"""Tests for the AgL type-checking pass.

All tests drive real AgL source through ``parse_program`` + ``resolve`` +
``check``, asserting on user-visible behavior: raised ``AglTypeError``
diagnostics and type-table / contract-spec observables via the public
``CheckedProgram`` API.

Grammar notes (shape constraints from the grammar):
- ``record`` uses indented field syntax: ``record Foo\\n  x: int``
- ``enum`` uses pipe variants: ``enum E\\n  | A(x: int)\\n  | B``
- ``raise`` can only appear at ``expr`` level (top-level block item, let/var
  rhs, funcdef body, lambda body) — NOT inline in branch bodies.
- Programs must not end in a ``let``/``var`` declaration.

Tests deliberately do *not* pin internal implementation details.
"""

from __future__ import annotations

from typing import cast

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.parser import parse_program
from agm.agl.scope import resolve
from agm.agl.scope.symbols import BinderKind, BindingRef, ScopeNode
from agm.agl.scope.symbols import ResolvedProgram as _ResolvedProgram
from agm.agl.semantics.type_table import TypeDef, TypeTable, comparable_types
from agm.agl.semantics.types import (
    BUILTIN_EXCEPTION_NAMES,
    BUILTIN_PRELUDE_TYPE_NAMES,
    BUILTIN_PRELUDE_TYPES,
    EXCEPTION_BASE,
    TypeVarType,
    is_assignable,
    is_json_shaped,
)
from agm.agl.syntax.nodes import (
    AgentDecl,
    AssignStmt,
    AssignTarget,
    Block,
    Call,
    Case,
    Cast,
    DictEntry,
    DictLit,
    Do,
    FieldAccess,
    FuncDef,
    If,
    IndexAccess,
    IndexTarget,
    IntLit,
    Item,
    Lambda,
    LetDecl,
    ListLit,
    NamedArg,
    Param,
    ParamDecl,
    ParamKind,
    Program,
    Raise,
    StringLit,
    Try,
    TypeApply,
    UnitLit,
    VarDecl,
    VarRef,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import (
    AgentT,
    BoolT,
    DictT,
    FuncT,
    IntT,
    JsonT,
    ListT,
    TextT,
    UnitT,
)
from agm.agl.typecheck import (
    AgentType,
    AglTypeError,
    BoolType,
    BottomType,
    CheckedProgram,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    FunctionSignature,
    FunctionType,
    IntType,
    JsonType,
    ListType,
    RecordType,
    TextType,
    TypeEnvironment,
    UnitType,
    check,
)
from agm.agl.typecheck.builder import _TypeBuilder
from agm.agl.typecheck.env import ConstructorSignature, GenericTypeDef
from tests._agl_helpers import all_node_ids

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_capabilities() -> HostCapabilities:
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


def text_only_caps() -> HostCapabilities:
    return HostCapabilities(
        agent_names=frozenset(),
        has_default_agent=True,
        codec_kinds={"text": frozenset({"text"})},
    )


def no_agent_caps() -> HostCapabilities:
    return HostCapabilities(
        agent_names=frozenset(),
        has_default_agent=False,
        supports_shell_exec=True,
        codec_kinds={
            "text": frozenset({"text"}),
            "json": frozenset(
                {"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}
            ),
        },
    )


def no_exec_caps() -> HostCapabilities:
    return HostCapabilities(
        agent_names=frozenset(),
        has_default_agent=True,
        supports_shell_exec=False,
        codec_kinds={"text": frozenset({"text"})},
    )


def parse_resolve_check(
    source: str, capabilities: HostCapabilities | None = None
) -> CheckedProgram:
    if capabilities is None:
        capabilities = default_capabilities()
    prog = parse_program(source)
    return check(resolve(prog), capabilities)


def accept_type(source: str, capabilities: HostCapabilities | None = None) -> CheckedProgram:
    return parse_resolve_check(source, capabilities)


def reject_type(source: str, capabilities: HostCapabilities | None = None) -> AglTypeError:
    """Assert that the source is rejected with an AglTypeError."""
    with pytest.raises(AglTypeError) as exc_info:
        parse_resolve_check(source, capabilities)
    return exc_info.value


def reject_any(source: str, capabilities: HostCapabilities | None = None) -> Exception:
    """Assert that the source is rejected with any kind of error."""
    with pytest.raises(Exception) as exc_info:
        parse_resolve_check(source, capabilities)
    return exc_info.value


def mk_span(line: int = 1, col: int = 1) -> SourceSpan:
    return SourceSpan(
        start_line=line, start_col=col, end_line=line, end_col=col + 1,
        start_offset=0, end_offset=1,
    )


_mk_node_id_counter = 100_000


def _mk_node_id() -> int:
    global _mk_node_id_counter
    _mk_node_id_counter += 1
    return _mk_node_id_counter


# ---------------------------------------------------------------------------
# Types: BottomType
# ---------------------------------------------------------------------------


class TestBottomType:
    def test_repr(self) -> None:
        assert repr(BottomType()) == "bottom"

    def test_kind(self) -> None:
        assert BottomType().kind == "bottom"

    def test_is_assignable_to_any(self) -> None:
        for t in (TextType(), IntType(), BoolType(), UnitType(), AgentType(), JsonType()):
            assert is_assignable(BottomType(), t)

    def test_not_json_shaped(self) -> None:
        assert not is_json_shaped(BottomType())

    def test_not_comparable_left(self) -> None:
        assert not comparable_types(BottomType(), IntType(), _EMPTY_TABLE)

    def test_not_comparable_right(self) -> None:
        assert not comparable_types(IntType(), BottomType(), _EMPTY_TABLE)

    def test_frozen(self) -> None:
        b = BottomType()
        with pytest.raises((AttributeError, TypeError)):
            setattr(b, "x", 1)

    def test_equality(self) -> None:
        assert BottomType() == BottomType()


# ---------------------------------------------------------------------------
# Types: is_json_shaped, comparable_types, is_assignable
# ---------------------------------------------------------------------------


class TestIsJsonShaped:
    def test_primitives(self) -> None:
        for t in (TextType(), JsonType(), BoolType(), IntType(), DecimalType()):
            assert is_json_shaped(t)

    def test_list_of_json_shaped(self) -> None:
        assert is_json_shaped(ListType(elem=IntType()))
        assert is_json_shaped(ListType(elem=JsonType()))

    def test_list_of_non_json(self) -> None:
        assert not is_json_shaped(ListType(elem=UnitType()))

    def test_dict_of_json_shaped(self) -> None:
        assert is_json_shaped(DictType(value=TextType()))

    def test_dict_of_non_json(self) -> None:
        assert not is_json_shaped(DictType(value=AgentType()))

    def test_record_not_json_shaped(self) -> None:
        rt = RecordType(name="R", fields={"x": IntType()})
        assert not is_json_shaped(rt)

    def test_enum_not_json_shaped(self) -> None:
        et = EnumType(name="E", variants={"A": {}})
        assert not is_json_shaped(et)

    def test_exception_not_json_shaped(self) -> None:
        assert not is_json_shaped(ExceptionType(name="Ex", fields={}))

    def test_unit_not_json_shaped(self) -> None:
        assert not is_json_shaped(UnitType())

    def test_agent_not_json_shaped(self) -> None:
        assert not is_json_shaped(AgentType())

    def test_function_not_json_shaped(self) -> None:
        assert not is_json_shaped(FunctionType(params=(IntType(),), result=IntType()))

    def test_bottom_not_json_shaped(self) -> None:
        assert not is_json_shaped(BottomType())


def _table_for(*nominal_types: RecordType | EnumType) -> TypeTable:
    """Build a ``TypeTable`` with a ``TypeDef`` registered for each nominal type.

    Lets ``comparable_types``' record/enum arms resolve field/variant shapes
    for ad hoc types constructed directly in these tests, rather than through
    a real program's type builder.
    """
    table = TypeTable()
    for t in nominal_types:
        if isinstance(t, RecordType):
            table.register(
                TypeDef(
                    kind="record",
                    name=t.name,
                    module_id=t.module_id,
                    fields=tuple(t.fields.items()),
                )
            )
        else:
            table.register(
                TypeDef(
                    kind="enum",
                    name=t.name,
                    module_id=t.module_id,
                    variants=tuple(
                        (vname, tuple(vfields.items())) for vname, vfields in t.variants.items()
                    ),
                )
            )
    return table


_EMPTY_TABLE = TypeTable()


class TestComparableTypes:
    def test_same_int(self) -> None:
        assert comparable_types(IntType(), IntType(), _EMPTY_TABLE)

    def test_same_text(self) -> None:
        assert comparable_types(TextType(), TextType(), _EMPTY_TABLE)

    def test_same_bool(self) -> None:
        assert comparable_types(BoolType(), BoolType(), _EMPTY_TABLE)

    def test_int_decimal_cross(self) -> None:
        assert comparable_types(IntType(), DecimalType(), _EMPTY_TABLE)
        assert comparable_types(DecimalType(), IntType(), _EMPTY_TABLE)

    def test_agent_not_comparable(self) -> None:
        assert not comparable_types(AgentType(), AgentType(), _EMPTY_TABLE)

    def test_function_not_comparable(self) -> None:
        ft = FunctionType(params=(), result=IntType())
        assert not comparable_types(ft, ft, _EMPTY_TABLE)

    def test_unit_not_comparable(self) -> None:
        assert not comparable_types(UnitType(), UnitType(), _EMPTY_TABLE)

    def test_bottom_not_comparable_left(self) -> None:
        assert not comparable_types(BottomType(), IntType(), _EMPTY_TABLE)

    def test_bottom_not_comparable_right(self) -> None:
        assert not comparable_types(IntType(), BottomType(), _EMPTY_TABLE)

    def test_cross_type_not_comparable(self) -> None:
        assert not comparable_types(TextType(), IntType(), _EMPTY_TABLE)
        assert not comparable_types(JsonType(), TextType(), _EMPTY_TABLE)

    # Transitive rejection: containers/records/enums holding non-comparable types.
    def test_list_of_function_not_comparable(self) -> None:
        ft = FunctionType(params=(), result=IntType())
        assert not comparable_types(ListType(elem=ft), ListType(elem=ft), _EMPTY_TABLE)

    def test_list_of_agent_not_comparable(self) -> None:
        assert not comparable_types(
            ListType(elem=AgentType()), ListType(elem=AgentType()), _EMPTY_TABLE
        )

    def test_list_of_unit_not_comparable(self) -> None:
        assert not comparable_types(
            ListType(elem=UnitType()), ListType(elem=UnitType()), _EMPTY_TABLE
        )

    def test_dict_of_function_not_comparable(self) -> None:
        ft = FunctionType(params=(IntType(),), result=TextType())
        assert not comparable_types(DictType(value=ft), DictType(value=ft), _EMPTY_TABLE)

    def test_record_with_agent_field_not_comparable(self) -> None:
        rt = RecordType(name="R", fields={"a": AgentType()})
        assert not comparable_types(rt, rt, _table_for(rt))

    def test_record_with_function_field_not_comparable(self) -> None:
        ft = FunctionType(params=(), result=IntType())
        rt = RecordType(name="R", fields={"f": ft})
        assert not comparable_types(rt, rt, _table_for(rt))

    def test_enum_with_function_field_not_comparable(self) -> None:
        ft = FunctionType(params=(), result=IntType())
        et = EnumType(name="E", variants={"A": {"fn": ft}})
        assert not comparable_types(et, et, _table_for(et))

    def test_exception_with_function_field_not_comparable(self) -> None:
        ft = FunctionType(params=(), result=IntType())
        et = ExceptionType(name="E", fields={"handler": ft})
        assert not comparable_types(et, et, _EMPTY_TABLE)

    def test_record_with_only_scalars_comparable(self) -> None:
        rt = RecordType(name="R", fields={"x": IntType(), "y": TextType()})
        assert comparable_types(rt, rt, _table_for(rt))

    def test_list_of_int_comparable(self) -> None:
        assert comparable_types(ListType(elem=IntType()), ListType(elem=IntType()), _EMPTY_TABLE)

    def test_dict_of_text_comparable(self) -> None:
        assert comparable_types(
            DictType(value=TextType()), DictType(value=TextType()), _EMPTY_TABLE
        )


class TestIsAssignable:
    def test_same_type(self) -> None:
        assert is_assignable(IntType(), IntType())
        assert is_assignable(TextType(), TextType())

    def test_int_to_decimal(self) -> None:
        assert is_assignable(IntType(), DecimalType())

    def test_decimal_to_int_not(self) -> None:
        assert not is_assignable(DecimalType(), IntType())

    def test_json_accepts_json_shaped(self) -> None:
        assert is_assignable(TextType(), JsonType())
        assert is_assignable(IntType(), JsonType())
        assert is_assignable(BoolType(), JsonType())
        assert is_assignable(DecimalType(), JsonType())
        assert is_assignable(JsonType(), JsonType())

    def test_json_rejects_non_shaped(self) -> None:
        rt = RecordType(name="R", fields={})
        assert not is_assignable(rt, JsonType())
        assert not is_assignable(UnitType(), JsonType())
        assert not is_assignable(AgentType(), JsonType())

    def test_bottom_to_any(self) -> None:
        assert is_assignable(BottomType(), IntType())
        assert is_assignable(BottomType(), TextType())
        assert is_assignable(BottomType(), FunctionType(params=(), result=IntType()))

    def test_function_exact_only(self) -> None:
        ft = FunctionType(params=(IntType(),), result=IntType())
        assert is_assignable(ft, ft)
        ft2 = FunctionType(params=(TextType(),), result=IntType())
        assert not is_assignable(ft, ft2)


# ---------------------------------------------------------------------------
# TypeEnvironment
# ---------------------------------------------------------------------------


class TestTypeEnvironment:
    def test_builtin_exceptions_available(self) -> None:
        env = TypeEnvironment()
        for name in BUILTIN_EXCEPTION_NAMES:
            assert env.has_type(name)
            t = env.get_type(name)
            assert isinstance(t, ExceptionType)

    def test_builtin_prelude_available(self) -> None:
        env = TypeEnvironment()
        for name in BUILTIN_PRELUDE_TYPE_NAMES:
            assert env.has_type(name)

    def test_register_type(self) -> None:
        env = TypeEnvironment()
        rt = RecordType(name="Foo", fields={})
        env.register_type("Foo", rt)
        assert env.get_type("Foo") == rt

    def test_register_alias(self) -> None:
        from agm.agl.syntax.types import NameT
        env = TypeEnvironment()
        sp = mk_span()
        env.register_alias("MyInt", IntT(span=sp, node_id=1))
        result = env.resolve_type_expr(NameT(name="MyInt", span=sp, node_id=2))
        assert result == IntType()

    def test_resolve_type_expr_text(self) -> None:
        env = TypeEnvironment()
        sp = mk_span()
        assert env.resolve_type_expr(TextT(span=sp, node_id=1)) == TextType()

    def test_resolve_type_expr_json(self) -> None:
        env = TypeEnvironment()
        sp = mk_span()
        assert env.resolve_type_expr(JsonT(span=sp, node_id=1)) == JsonType()

    def test_resolve_type_expr_bool(self) -> None:
        env = TypeEnvironment()
        sp = mk_span()
        assert env.resolve_type_expr(BoolT(span=sp, node_id=1)) == BoolType()

    def test_resolve_type_expr_int(self) -> None:
        env = TypeEnvironment()
        sp = mk_span()
        assert env.resolve_type_expr(IntT(span=sp, node_id=1)) == IntType()

    def test_resolve_type_expr_decimal(self) -> None:
        env = TypeEnvironment()
        sp = mk_span()
        from agm.agl.syntax.types import DecimalT
        assert env.resolve_type_expr(DecimalT(span=sp, node_id=1)) == DecimalType()

    def test_resolve_type_expr_unit(self) -> None:
        env = TypeEnvironment()
        sp = mk_span()
        assert env.resolve_type_expr(UnitT(span=sp, node_id=1)) == UnitType()

    def test_resolve_type_expr_agent(self) -> None:
        env = TypeEnvironment()
        sp = mk_span()
        assert env.resolve_type_expr(AgentT(span=sp, node_id=1)) == AgentType()

    def test_resolve_list_type(self) -> None:
        env = TypeEnvironment()
        sp = mk_span()
        result = env.resolve_type_expr(ListT(elem=IntT(span=sp, node_id=1), span=sp, node_id=2))
        assert result == ListType(elem=IntType())

    def test_resolve_dict_type(self) -> None:
        env = TypeEnvironment()
        sp = mk_span()
        result = env.resolve_type_expr(
            DictT(value=TextT(span=sp, node_id=1), span=sp, node_id=2)
        )
        assert result == DictType(value=TextType())

    def test_resolve_func_type(self) -> None:
        env = TypeEnvironment()
        sp = mk_span()
        result = env.resolve_type_expr(
            FuncT(
                params=(IntT(span=sp, node_id=1),),
                result=TextT(span=sp, node_id=2),
                span=sp,
                node_id=3,
            )
        )
        assert result == FunctionType(params=(IntType(),), result=TextType())

    def test_resolve_unknown_name_raises(self) -> None:
        from agm.agl.syntax.types import NameT
        env = TypeEnvironment()
        with pytest.raises(AglTypeError, match="Unknown type"):
            env.resolve_type_expr(NameT(name="NonExistent", span=mk_span(), node_id=1))

    def test_alias_cycle_detected(self) -> None:
        from agm.agl.syntax.types import NameT
        env = TypeEnvironment()
        sp = mk_span()
        env.register_alias("A", NameT(name="B", span=sp, node_id=1))
        env.register_alias("B", NameT(name="A", span=sp, node_id=2))
        with pytest.raises(AglTypeError, match="cycle"):
            env.resolve_type_expr(NameT(name="A", span=sp, node_id=3))

    def test_binding_type_roundtrip(self) -> None:
        env = TypeEnvironment()
        env.set_binding_type(42, IntType())
        assert env.get_binding_type(42) == IntType()

    def test_function_signature_roundtrip(self) -> None:
        env = TypeEnvironment()
        sig = FunctionSignature(
            params=(("x", IntType(), False),),
            result=TextType(),
        )
        env.register_function_signature("f", sig)
        assert env.get_function_signature("f") == sig
        assert env.get_function_signature("g") is None

    def test_all_function_signatures(self) -> None:
        env = TypeEnvironment()
        sig = FunctionSignature(params=(), result=UnitType())
        env.register_function_signature("g", sig)
        sigs = env.all_function_signatures()
        assert "g" in sigs

    def test_seed_from_copies_types_and_bindings(self) -> None:
        env1 = TypeEnvironment()
        rt = RecordType(name="Foo", fields={})
        env1.register_type("Foo", rt)
        env1.set_binding_type(99, IntType())
        sig = FunctionSignature(params=(), result=UnitType())
        env1.register_function_signature("h", sig)

        env2 = TypeEnvironment()
        env2.seed_from(env1)
        assert env2.get_type("Foo") == rt
        assert env2.get_binding_type(99) == IntType()
        assert env2.get_function_signature("h") == sig

    def test_seed_preserves_own_builtins(self) -> None:
        env1 = TypeEnvironment()
        env2 = TypeEnvironment()
        env2.seed_from(env1)
        assert env2.has_type("Abort")

    def test_unregister_name(self) -> None:
        env = TypeEnvironment()
        env.register_type("Foo", RecordType(name="Foo", fields={}))
        env.unregister_name("Foo")
        assert env.get_type("Foo") is None

    def test_unregister_builtin_is_noop(self) -> None:
        env = TypeEnvironment()
        env.unregister_name("Abort")
        assert env.has_type("Abort")

    def test_all_declared_type_names(self) -> None:
        env = TypeEnvironment()
        env.register_type("MyRec", RecordType(name="MyRec", fields={}))
        names = env.all_declared_type_names()
        assert "MyRec" in names
        assert "Abort" in names

    def test_resolve_named_type(self) -> None:
        env = TypeEnvironment()
        rt = RecordType(name="R", fields={})
        env.register_type("R", rt)
        result = env.resolve_named_type("R")
        assert result == rt

    def test_resolve_named_type_unknown(self) -> None:
        env = TypeEnvironment()
        assert env.resolve_named_type("Unknown") is None

    def test_resolve_named_type_multiple_candidates_returns_none(self) -> None:
        # Coverage: resolve_named_type returns None when multiple candidates exist.
        # Two unqualified imports of the same name → ambiguous → return None.
        from agm.agl.modules.ids import ModuleId
        from agm.agl.scope.imports import ImportEnv
        mod_a = ModuleId.from_dotted("moda")
        mod_b = ModuleId.from_dotted("modb")
        color_a = RecordType(name="Color", fields={})
        color_b = RecordType(name="Color", fields={})
        graph_table: dict[tuple[ModuleId, str], RecordType] = {
            (mod_a, "Color"): color_a,
            (mod_b, "Color"): color_b,
        }
        # Both modules expose "Color" unqualified.
        unqualified: dict[str, frozenset[tuple[ModuleId, str]]] = {
            "Color": frozenset({(mod_a, "Color"), (mod_b, "Color")}),
        }
        import_env = ImportEnv(unqualified=unqualified, qualified={})
        env = TypeEnvironment(graph_type_table=graph_table, import_env=import_env)
        # Both entries are in graph_table → type_candidates has 2 elements → False branch.
        result = env.resolve_named_type("Color")
        assert result is None

    def test_resolve_type_by_module_id_no_graph_table(self) -> None:
        # Coverage: resolve_type_by_module_id returns None in single-program mode (no graph table).
        from agm.agl.modules.ids import ModuleId
        env = TypeEnvironment()  # no graph_type_table
        result = env.resolve_type_by_module_id(ModuleId.from_dotted("mymod"), "Color")
        assert result is None

    def test_get_generic_type_from_module_no_graph_table(self) -> None:
        # Coverage: env.py get_generic_type_from_module — single-program mode returns None.
        from agm.agl.modules.ids import ModuleId
        env = TypeEnvironment()  # no graph_generic_table
        result = env.get_generic_type_from_module(ModuleId.from_dotted("lib"), "Box")
        assert result is None

    def test_get_open_imported_type_no_graph_context(self) -> None:
        env = TypeEnvironment()
        assert env.get_open_imported_type("Point") is None

    def test_get_open_imported_type_ignores_missing_graph_entries(self) -> None:
        from agm.agl.modules.ids import ModuleId
        from agm.agl.scope.imports import ImportEnv

        mod = ModuleId.from_dotted("lib")
        import_env = ImportEnv(
            unqualified={"Point": frozenset({(mod, "Point")})},
            qualified={},
        )
        env = TypeEnvironment(graph_type_table={}, import_env=import_env)
        assert env.get_open_imported_type("Point") is None

    def test_get_open_imported_type_ambiguous_returns_none(self) -> None:
        from agm.agl.modules.ids import ModuleId
        from agm.agl.scope.imports import ImportEnv

        mod_a = ModuleId.from_dotted("a")
        mod_b = ModuleId.from_dotted("b")
        graph_table = {
            (mod_a, "Point"): RecordType(name="Point", fields={}),
            (mod_b, "Point"): RecordType(name="Point", fields={}),
        }
        import_env = ImportEnv(
            unqualified={"Point": frozenset({(mod_a, "Point"), (mod_b, "Point")})},
            qualified={},
        )
        env = TypeEnvironment(graph_type_table=graph_table, import_env=import_env)
        assert env.get_open_imported_type("Point") is None

    def test_get_ctor_sig_from_module_no_graph_table(self) -> None:
        # Coverage: env.py get_ctor_sig_from_module — single-program mode returns None.
        from agm.agl.modules.ids import ModuleId
        env = TypeEnvironment()  # no graph_ctor_sig_table
        result = env.get_ctor_sig_from_module(ModuleId.from_dotted("lib"), "Box", None)
        assert result is None

    def test_resolve_unknown_type_expr_kind_raises(self) -> None:
        env = TypeEnvironment()
        with pytest.raises(AglTypeError, match="Unknown type expression"):
            env.resolve_type_expr("not_a_type_expr")

    def test_resolve_binding_none_for_missing(self) -> None:
        env = TypeEnvironment()
        ref = BindingRef(
            name="x", mutable=False, decl_span=mk_span(), decl_node_id=999,
            kind=BinderKind.let_binding,
        )
        assert env.resolve_binding(ref) is None


# ---------------------------------------------------------------------------
# Block typing
# ---------------------------------------------------------------------------


class TestBlockTyping:
    def test_block_last_expr_is_block_type(self) -> None:
        r = accept_type("let x = 1\nx")
        assert r.resolved.program is not None

    def test_block_ending_in_let_is_error(self) -> None:
        err = reject_type("let x = 1")
        assert "let" in str(err) or "var" in str(err) or "declaration" in str(err)

    def test_block_ending_in_var_is_error(self) -> None:
        err = reject_type("var x = 1")
        assert "let" in str(err) or "var" in str(err) or "declaration" in str(err)

    def test_assign_at_end_is_valid(self) -> None:
        # AssignStmt at end is valid (it produces unit, not a LetDecl)
        r = accept_type("var x = 1\nx := 2")
        assert r.resolved.program is not None

    def test_funcdef_at_end_is_ok(self) -> None:
        # FuncDef at end is a declaration, not LetDecl/VarDecl — OK
        r = accept_type("def f() -> int = 1")
        assert r.resolved.program is not None

    def test_unit_literal_valid(self) -> None:
        r = accept_type("()")
        assert r.resolved.program is not None

    def test_let_followed_by_expr(self) -> None:
        r = accept_type("let x = 1\nlet y = 2\nx + y")
        assert r.resolved.program is not None

    def test_param_declaration(self) -> None:
        r = accept_type("param x\nx")
        assert r.resolved.program is not None

    def test_param_with_annotation(self) -> None:
        r = accept_type("param n: int\nn")
        assert r.resolved.program is not None

    def test_param_defaults_to_text(self) -> None:
        r = accept_type("param x\nx")
        prog = r.resolved.program
        param_decl = prog.body.items[0]
        assert isinstance(param_decl, ParamDecl)
        binding_type = r.type_env.get_binding_type(param_decl.node_id)
        assert binding_type == TextType()


# ---------------------------------------------------------------------------
# Unit type propagation
# ---------------------------------------------------------------------------


class TestUnitPropagation:
    def test_print_yields_unit(self) -> None:
        r = accept_type("let u: unit = print(42)\nu")
        prog = r.resolved.program
        let_decl = prog.body.items[0]
        assert isinstance(let_decl, LetDecl)
        assert r.type_env.get_binding_type(let_decl.node_id) == UnitType()

    def test_if_no_else_yields_unit(self) -> None:
        r = accept_type("let u: unit = if true => ()\nu")
        let_decl = r.resolved.program.body.items[0]
        assert isinstance(let_decl, LetDecl)
        assert r.type_env.get_binding_type(let_decl.node_id) == UnitType()

    def test_if_no_else_branch_body_must_be_unit(self) -> None:
        err = reject_type("if true => 1\n()")
        assert "unit" in str(err).lower()

    def test_unit_lit(self) -> None:
        r = accept_type("let u: unit = ()\nu")
        let_decl = r.resolved.program.body.items[0]
        assert isinstance(let_decl, LetDecl)
        assert r.type_env.get_binding_type(let_decl.node_id) == UnitType()

    def test_agent_decl_yields_agent_type(self) -> None:
        r = accept_type("agent reviewer\nreviewer")
        prog = r.resolved.program
        agent_decl = prog.body.items[0]
        assert isinstance(agent_decl, AgentDecl)
        assert r.type_env.get_binding_type(agent_decl.node_id) == AgentType()

    def test_assign_is_valid_block_item(self) -> None:
        r = accept_type("var x = 1\nx := 2\n()")
        assert r.resolved.program is not None


# ---------------------------------------------------------------------------
# Literals
# ---------------------------------------------------------------------------


class TestLiterals:
    def test_int_literal(self) -> None:
        r = accept_type("let x = 42\nx")
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        assert r.type_env.get_binding_type(decl.node_id) == IntType()

    def test_decimal_literal(self) -> None:
        r = accept_type("let x = 3.14\nx")
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        assert r.type_env.get_binding_type(decl.node_id) == DecimalType()

    def test_bool_literal(self) -> None:
        r = accept_type("let x = true\nx")
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        assert r.type_env.get_binding_type(decl.node_id) == BoolType()

    def test_null_is_json(self) -> None:
        r = accept_type("let x: json = null\nx")
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        assert r.type_env.get_binding_type(decl.node_id) == JsonType()

    def test_string_is_text(self) -> None:
        r = accept_type('let x = "hello"\nx')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        assert r.type_env.get_binding_type(decl.node_id) == TextType()

    def test_int_widens_to_decimal_annotation(self) -> None:
        r = accept_type("let d: decimal = 3\nd")
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        assert r.type_env.get_binding_type(decl.node_id) == DecimalType()

    def test_mismatch_text_vs_int(self) -> None:
        err = reject_type("let x: text = 42\nx")
        assert "mismatch" in str(err).lower() or "expected" in str(err).lower()

    def test_mismatch_int_vs_bool(self) -> None:
        err = reject_type("let x: bool = 42\nx")
        assert "bool" in str(err).lower()
        assert "int" in str(err).lower()

    def test_mismatch_text_vs_bool(self) -> None:
        err = reject_type('let x: bool = "hello"\nx')
        assert "bool" in str(err).lower()
        assert "text" in str(err).lower()


# ---------------------------------------------------------------------------
# print() builtin
# ---------------------------------------------------------------------------


class TestPrint:
    def test_print_int(self) -> None:
        accept_type("print(42)")

    def test_print_text(self) -> None:
        accept_type('print("hello")')

    def test_print_bool(self) -> None:
        accept_type("print(true)")

    def test_print_null(self) -> None:
        accept_type("print(null)")

    def test_print_unit(self) -> None:
        accept_type("print(())")

    def test_print_function_accepted(self) -> None:
        accept_type("let f = fn(x: int) -> int => x\nprint(f)")

    def test_print_agent_accepted(self) -> None:
        accept_type("agent a\nprint(a)")

    def test_print_wrong_arg_count(self) -> None:
        err = reject_type("print(1, 2)")
        assert "print" in str(err).lower() or "argument" in str(err).lower()

    def test_print_no_args(self) -> None:
        err = reject_type("print()")
        assert "print" in str(err).lower() or "argument" in str(err).lower()

    def test_print_named_arg(self) -> None:
        err = reject_type("print(x = 42)")
        assert "print" in str(err).lower() or "argument" in str(err).lower()


# ---------------------------------------------------------------------------
# render() builtin
# ---------------------------------------------------------------------------


class TestRenderBuiltin:
    def test_render_returns_text(self) -> None:
        r = accept_type("let s: text = render([1, 2])\ns")
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        assert r.type_env.get_binding_type(decl.node_id) == TextType()

    def test_render_accepts_options(self) -> None:
        accept_type('render("hello", pretty = false, quote_strings = false)')

    def test_render_wrong_arg_count_rejected(self) -> None:
        err = reject_type("render(1, 2)")
        assert "render" in str(err).lower()
        assert "one positional" in str(err).lower()

    def test_render_unknown_named_arg_rejected(self) -> None:
        err = reject_type("render(1, style = true)")
        assert "render" in str(err).lower()
        assert "unknown" in str(err).lower()

    def test_render_option_must_be_bool(self) -> None:
        err = reject_type('render(1, pretty = "yes")')
        assert "bool" in str(err).lower()


# ---------------------------------------------------------------------------
# ask() builtin
# ---------------------------------------------------------------------------


class TestAsk:
    def test_ask_text_default(self) -> None:
        r = accept_type('ask("hello")')
        call = r.resolved.program.body.items[0]
        assert isinstance(call, Call)
        spec = r.contract_specs[call.node_id]
        assert spec.target_type == TextType()
        assert spec.codec_name == "text"
        assert spec.strict_json is None

    def test_ask_with_annotation_int(self) -> None:
        # block must end with trailing expr
        r = accept_type('let n: int = ask("Q")\nn')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        call = decl.value
        assert isinstance(call, Call)
        spec = r.contract_specs[call.node_id]
        assert spec.target_type == IntType()
        assert spec.codec_name == "json"

    def test_ask_with_unit_target_has_no_contract(self) -> None:
        r = accept_type('let result: unit = ask("Q")\nresult')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        assert decl.value.node_id not in r.contract_specs
        assert r.node_types[decl.value.node_id] == UnitType()

    def test_ask_with_explicit_agent(self) -> None:
        r = accept_type('agent reviewer\nask("Q", agent = reviewer)')
        assert r.resolved.program is not None

    def test_ask_no_default_agent_raises(self) -> None:
        err = reject_type('ask("Q")', capabilities=no_agent_caps())
        assert "agent" in str(err).lower() or "default" in str(err).lower()

    def test_ask_no_prompt_raises(self) -> None:
        err = reject_type("ask()")
        assert "prompt" in str(err).lower() or "argument" in str(err).lower()

    def test_ask_wrong_agent_type(self) -> None:
        err = reject_type('let x = "not_agent"\nask("Q", agent = x)')
        assert "agent" in str(err).lower()

    def test_ask_with_json_codec(self) -> None:
        r = accept_type('let n: int = ask("Q", format = "json")\nn')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        spec = r.contract_specs[decl.value.node_id]
        assert spec.codec_name == "json"

    def test_ask_strict_json_true(self) -> None:
        r = accept_type('let n: int = ask("Q", format = "json", strict_json = true)\nn')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        spec = r.contract_specs[decl.value.node_id]
        assert spec.strict_json is True

    def test_ask_strict_json_without_json_codec_raises(self) -> None:
        err = reject_type('let x = ask("Q", strict_json = true)\nx')
        assert "strict_json" in str(err).lower() or "json" in str(err).lower()

    def test_ask_format_non_string_raises(self) -> None:
        err = reject_type('let n: int = ask("Q", format = 42)\nn')
        assert "format" in str(err).lower() or "static" in str(err).lower()

    def test_ask_strict_json_non_bool_raises(self) -> None:
        err = reject_type('let n: int = ask("Q", format = "json", strict_json = "yes")\nn')
        assert "strict_json" in str(err).lower() or "bool" in str(err).lower()

    def test_ask_on_parse_error_abort(self) -> None:
        r = accept_type('let n: int = ask("Q", on_parse_error = Abort())\nn')
        assert r.call_sites[0].parse_policy == "abort"

    def test_ask_on_parse_error_retry(self) -> None:
        r = accept_type('let n: int = ask("Q", on_parse_error = Retry(n = 3))\nn')
        assert r.call_sites[0].parse_policy == "retry[3]"

    def test_ask_on_parse_error_text_warns(self) -> None:
        r = accept_type('ask("Q", on_parse_error = Abort())')
        assert len(r.warnings) == 1
        assert "on_parse_error" in r.warnings[0].message

    def test_ask_on_parse_error_bare_abort_varref(self) -> None:
        # Bare ``Abort`` (no parens) is accepted as abort policy.
        r = accept_type('let n: int = ask("Q", on_parse_error = Abort)\nn')
        assert r.call_sites[0].parse_policy == "abort"

    def test_ask_on_parse_error_bare_qualified_abort(self) -> None:
        # Bare ``ParsePolicy.Abort`` (no parens) is accepted as abort policy.
        r = accept_type('let n: int = ask("Q", on_parse_error = ParsePolicy.Abort)\nn')
        assert r.call_sites[0].parse_policy == "abort"

    def test_ask_on_parse_error_bad_qualified_policy_raises(self) -> None:
        # A FieldAccess callee with wrong qualifier is rejected.
        err = reject_type(
            "enum FooBar\n  | Abort\n"
            'let n: int = ask("Q", on_parse_error = FooBar.Abort())\nn'
        )
        assert "parse_error" in str(err).lower() or "ParsePolicy" in str(err)

    def test_ask_on_parse_error_bare_wrong_qualifier_raises(self) -> None:
        # Bare FieldAccess ``SomethingElse.Abort`` (no parens, non-ParsePolicy qualifier)
        # is rejected even though the field name is "Abort".
        err = reject_type(
            "enum SomethingElse\n  | Abort\n"
            'let n: int = ask("Q", on_parse_error = SomethingElse.Abort)\nn'
        )
        assert "parse_error" in str(err).lower() or "ParsePolicy" in str(err)

    def test_ask_function_target_rejected(self) -> None:
        err = reject_type('let f: (int) -> int = ask("Q")\nf(1)')
        assert "function" in str(err).lower() or "agent" in str(err).lower()

    def test_ask_agent_target_rejected(self) -> None:
        err = reject_type('let a: agent = ask("Q")\na')
        assert "function" in str(err).lower() or "agent" in str(err).lower()

    def test_ask_call_site_record(self) -> None:
        r = accept_type('ask("hello")')
        assert len(r.call_sites) == 1
        cs = r.call_sites[0]
        assert cs.callee == "ask"
        assert cs.parse_policy == "default"
        assert cs.line == 1

    def test_ask_unknown_codec_raises(self) -> None:
        err = reject_type('let x = ask("Q", format = "cbor")\nx')
        assert "cbor" in str(err) or "codec" in str(err).lower()

    def test_ask_codec_mismatch_raises(self) -> None:
        # text codec doesn't support int kind
        err = reject_type('let n: int = ask("Q", format = "text")\nn')
        assert "text" in str(err) or "codec" in str(err).lower() or "support" in str(err).lower()

    def test_ask_strict_json_false(self) -> None:
        r = accept_type('let n: int = ask("Q", format = "json", strict_json = false)\nn')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        spec = r.contract_specs[decl.value.node_id]
        assert spec.strict_json is False

    def test_ask_with_record_target_uses_json_codec(self) -> None:
        r = accept_type("record R\n  x: int\nlet r: R = ask(\"Q\")\nr")
        decl = r.resolved.program.body.items[1]
        assert isinstance(decl, LetDecl)
        spec = r.contract_specs[decl.value.node_id]
        assert spec.codec_name == "json"
        assert spec.target_type == r.type_env.get_type("R")


# ---------------------------------------------------------------------------
# ask-request() builtin
# ---------------------------------------------------------------------------


class TestAskRequest:
    """``ask-request`` is the side-effect-free twin of ``ask``: it builds the
    ``AgentRequest`` that ``ask`` would dispatch, without invoking the agent."""

    def test_default_target_is_text(self) -> None:
        r = accept_type('ask-request("Q")')
        call = r.resolved.program.body.items[0]
        assert isinstance(call, Call)
        spec = r.contract_specs[call.node_id]
        assert spec.target_type == TextType()
        assert spec.codec_name == "text"

    def test_explicit_type_arg_drives_contract(self) -> None:
        r = accept_type("record R\n  x: int\nask-request::[R](\"Q\")")
        call = r.resolved.program.body.items[1]
        assert isinstance(call, Call)
        spec = r.contract_specs[call.node_id]
        assert spec.codec_name == "json"
        assert spec.target_type == r.type_env.get_type("R")

    def test_unit_target_has_no_contract(self) -> None:
        r = accept_type('ask-request::[unit]("Q")')
        call = r.resolved.program.body.items[0]
        assert isinstance(call, Call)
        assert call.node_id not in r.contract_specs
        option_type = r.type_env.get_type("OutputContractOption")
        assert isinstance(option_type, EnumType)
        assert option_type.variants == {
            "None": {},
            "Some": {"value": r.type_env.get_type("OutputContract")},
        }

    @pytest.mark.parametrize(
        "option",
        ('format = "text"', "strict_json = true", "on_parse_error = Abort()"),
    )
    def test_unit_target_rejects_parse_options(self, option: str) -> None:
        err = reject_type(f'ask-request::[unit]("Q", {option})')
        assert "unit" in str(err)
        assert "no output contract" in str(err)

    def test_returns_agent_request_type(self) -> None:
        r = accept_type('let r = ask-request::[text]("Q")\nr')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        binding_type = r.type_env.get_binding_type(decl.node_id)
        assert binding_type == r.type_env.get_type("AgentRequest")

    def test_contextual_expected_type_ignored(self) -> None:
        # Unlike ``ask``, the target type is NOT inferred from context: a
        # contextual ``AgentRequest`` annotation does NOT make ask-request target
        # AgentRequest; the explicit ``::[int]`` drives the contract instead.
        r = accept_type('let r: AgentRequest = ask-request::[int]("Q")\nr')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        call = decl.value
        assert isinstance(call, Call)
        spec = r.contract_specs[call.node_id]
        assert spec.target_type == IntType()
        assert spec.codec_name == "json"
        assert r.node_types[call.node_id] == r.type_env.get_type("AgentRequest")

    def test_no_prompt_raises(self) -> None:
        err = reject_type("ask-request::[text]()")
        assert "prompt" in str(err).lower() or "argument" in str(err).lower()

    def test_too_many_positional_raises(self) -> None:
        err = reject_type('ask-request::[text]("a", "b")')
        assert "positional" in str(err).lower() or "argument" in str(err).lower()

    def test_too_many_type_args_raises(self) -> None:
        # ask-request with more than one explicit type argument is rejected.
        err = reject_type('ask-request::[int, text]("Q")')
        assert "type argument" in str(err).lower() or "got 2" in str(err)

    def test_unknown_named_arg_raises(self) -> None:
        err = reject_type('ask-request::[text]("Q", bogus = 1)')
        assert "bogus" in str(err) or "argument" in str(err).lower()

    def test_function_target_rejected(self) -> None:
        err = reject_type('ask-request::[(int) -> int]("Q")')
        assert "function" in str(err).lower() or "agent" in str(err).lower()

    def test_agent_target_rejected(self) -> None:
        err = reject_type('ask-request::[agent]("Q")')
        assert "function" in str(err).lower() or "agent" in str(err).lower()

    def test_with_explicit_agent(self) -> None:
        r = accept_type('agent reviewer\nask-request::[text]("Q", agent = reviewer)')
        assert r.resolved.program is not None

    def test_wrong_agent_type_raises(self) -> None:
        err = reject_type('let x = "no"\nask-request::[text]("Q", agent = x)')
        assert "agent" in str(err).lower()

    def test_strict_json_option(self) -> None:
        r = accept_type('ask-request::[int]("Q", format = "json", strict_json = true)')
        call = r.resolved.program.body.items[0]
        assert isinstance(call, Call)
        spec = r.contract_specs[call.node_id]
        assert spec.strict_json is True

    def test_on_parse_error_policy_recorded(self) -> None:
        r = accept_type('ask-request::[int]("Q", on_parse_error = Retry(n = 3))')
        assert len(r.call_sites) == 1
        cs = r.call_sites[0]
        assert cs.callee == "ask-request"
        assert cs.parse_policy == "retry[3]"

    def test_call_site_record(self) -> None:
        r = accept_type('ask-request::[text]("Q")')
        assert len(r.call_sites) == 1
        cs = r.call_sites[0]
        assert cs.callee == "ask-request"
        assert cs.parse_policy == "default"
        assert cs.line == 1

    def test_unknown_type_in_type_arg_raises(self) -> None:
        err = reject_type('ask-request::[NoSuchType]("Q")')
        assert "unknown type" in str(err).lower()

    def test_does_not_require_default_agent(self) -> None:
        # ask-request never dispatches, so it works without a default agent.
        r = accept_type('ask-request::[text]("Q")', capabilities=no_agent_caps())
        call = r.resolved.program.body.items[0]
        assert isinstance(call, Call)
        assert r.contract_specs[call.node_id].target_type == TextType()


# ---------------------------------------------------------------------------
# exec() builtin
# ---------------------------------------------------------------------------


class TestExec:
    def test_exec_rejected_without_shell_support(self) -> None:
        err = reject_type('exec("ls")', capabilities=no_exec_caps())
        assert "exec" in str(err).lower() or "shell" in str(err).lower()

    def test_exec_text_default(self) -> None:
        # exec without annotation → ExecResult (structured) form by default
        r = accept_type('let x = exec("ls")\nx')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        spec = r.contract_specs[decl.value.node_id]
        assert spec.structured_exec is True

    def test_exec_with_text_annotation(self) -> None:
        r = accept_type('let x: text = exec("ls")\nx')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        spec = r.contract_specs[decl.value.node_id]
        assert spec.target_type == TextType()
        assert spec.codec_name == "text"

    def test_exec_function_target_rejected(self) -> None:
        err = reject_type('let f: (int) -> int = exec("ls")\nf(1)')
        assert "function" in str(err).lower() or "agent" in str(err).lower()

    def test_exec_no_command_raises(self) -> None:
        err = reject_type("exec()")
        assert "command" in str(err).lower() or "argument" in str(err).lower()

    def test_exec_with_format_json(self) -> None:
        r = accept_type('let n: int = exec("ls", format = "json")\nn')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        spec = r.contract_specs[decl.value.node_id]
        assert spec.codec_name == "json"

    def test_exec_strict_json(self) -> None:
        r = accept_type('let n: int = exec("ls", format = "json", strict_json = true)\nn')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        spec = r.contract_specs[decl.value.node_id]
        assert spec.strict_json is True

    def test_exec_on_parse_error_text_warns(self) -> None:
        r = accept_type('let x: text = exec("ls", on_parse_error = Abort())\nx')
        assert len(r.warnings) == 1
        assert "on_parse_error" in r.warnings[0].message

    def test_exec_call_site_record(self) -> None:
        r = accept_type('exec("ls")')
        assert len(r.call_sites) == 1
        cs = r.call_sites[0]
        assert cs.callee == "exec"

    def test_exec_call_site_abort_policy(self) -> None:
        r = accept_type('let n: int = exec("ls", on_parse_error = Abort())\nn')
        assert r.call_sites[0].parse_policy == "abort"

    def test_exec_call_site_retry_policy(self) -> None:
        r = accept_type('let n: int = exec("ls", on_parse_error = Retry(n = 2))\nn')
        assert r.call_sites[0].parse_policy == "retry[2]"

    def test_exec_strict_json_without_json_raises(self) -> None:
        err = reject_type('let x: text = exec("ls", strict_json = true)\nx')
        assert "strict_json" in str(err).lower() or "json" in str(err).lower()

    def test_exec_format_non_string_raises(self) -> None:
        err = reject_type('let n: int = exec("ls", format = 42)\nn')
        assert "format" in str(err).lower() or "static" in str(err).lower()


# ---------------------------------------------------------------------------
# FuncDef declarations
# ---------------------------------------------------------------------------


class TestFuncDef:
    def test_simple_funcdef(self) -> None:
        r = accept_type("def f(x: int) -> int = x\nf(1)")
        assert "f" in r.function_signatures

    def test_funcdef_return_type_checked(self) -> None:
        err = reject_type("def f(x: int) -> text = x")
        assert "mismatch" in str(err).lower() or "expected" in str(err).lower()

    def test_funcdef_without_return_type_inferred(self) -> None:
        r = accept_type("def f(x: int) = x\nf")
        prog = r.resolved.program
        f_ref = prog.body.items[1]
        assert isinstance(f_ref, VarRef)
        t = r.node_types[f_ref.node_id]
        assert t == FunctionType(params=(IntType(),), result=IntType())

    def test_funcdef_without_return_type_checks_default_param(self) -> None:
        r = accept_type("def f(x: int = 1) = x\nf()")
        assert "f" in r.function_signatures

    def test_funcdef_without_return_type_body_always_raises(self) -> None:
        err = reject_type('def f() = raise Abort(message = "x")')
        msg = str(err).lower()
        assert "infer" in msg
        assert "return type" in msg

    def test_funcdef_without_return_type_recursive_call_needs_annotation(self) -> None:
        err = reject_type("def f(x: int) = f(x)")
        msg = str(err).lower()
        assert "infer" in msg
        assert "return type" in msg
        assert "annotation" in msg

    def test_funcdef_with_default_param(self) -> None:
        r = accept_type("def f(x: int, y: int = 0) -> int = x + y\nf(1)")
        assert "f" in r.function_signatures
        sig = r.function_signatures["f"]
        assert sig.params[1].has_default is True  # y has a default

    def test_funcdef_required_after_defaulted_raises(self) -> None:
        err = reject_type("def f(x: int = 0, y: int) -> int = x\nf(1, 2)")
        assert "default" in str(err).lower() or "required" in str(err).lower()

    def test_funcdef_callable(self) -> None:
        r = accept_type("def double(x: int) -> int = x * 2\ndouble(5)")
        assert r.resolved.program is not None

    def test_funcdef_return_type_is_function_type(self) -> None:
        r = accept_type("def f(x: int) -> int = x\nf")
        prog = r.resolved.program
        f_ref = prog.body.items[1]
        assert isinstance(f_ref, VarRef)
        t = r.node_types[f_ref.node_id]
        assert t == FunctionType(params=(IntType(),), result=IntType())

    def test_funcdef_with_raise_body(self) -> None:
        # raise has BottomType, assignable to any declared return type
        r = accept_type('def f(x: int) -> text = raise Abort(message = "err")\nf(1)')
        assert r.resolved.program is not None

    def test_funcdef_called_with_named_args(self) -> None:
        r = accept_type("def f(x: int, y: int = 0) -> int = x + y\nf(1, y = 2)")
        assert r.resolved.program is not None

    def test_funcdef_missing_required_arg(self) -> None:
        err = reject_type("def f(x: int, y: int) -> int = x + y\nf(1)")
        assert "missing" in str(err).lower() or "required" in str(err).lower()

    def test_funcdef_too_many_positional_args(self) -> None:
        err = reject_type("def f(x: int) -> int = x\nf(1, 2)")
        assert "too many" in str(err).lower() or "argument" in str(err).lower()

    def test_funcdef_unknown_named_arg(self) -> None:
        err = reject_type("def f(x: int) -> int = x\nf(z = 1)")
        assert "unknown" in str(err).lower() or "parameter" in str(err).lower()

    def test_funcdef_param_supplied_positionally_and_by_name(self) -> None:
        err = reject_type("def f(x: int) -> int = x\nf(1, x = 2)")
        assert (
            "positionally" in str(err).lower()
            or "both" in str(err).lower()
            or "duplicate" in str(err).lower()
        )

    def test_funcdef_default_wrong_type_raises(self) -> None:
        err = reject_type('def f(x: int = "bad") -> int = x')
        assert "mismatch" in str(err).lower() or "expected" in str(err).lower()

    def test_funcdef_all_named_args(self) -> None:
        r = accept_type("def f(x: int, y: int) -> int = x + y\nf(x = 1, y = 2)")
        assert r.resolved.program is not None

    def test_funcdef_named_then_positional(self) -> None:
        # Named args after positionals — checker must handle ordering
        r = accept_type("def f(x: int, y: int = 0) -> int = x + y\nf(1)")
        assert r.resolved.program is not None

    def test_funcdef_duplicate_named_arg_via_call(self) -> None:
        # The parser catches duplicate args at parse time (AglSyntaxError)
        # so we test via reject_any instead of reject_type
        err = reject_any("def f(x: int, y: int) -> int = x\nf(y = 1, y = 2)")
        assert "y" in str(err) or "duplicate" in str(err).lower()

    def test_funcdef_shadows_builtin_name_is_scope_error(self) -> None:
        # Scope pass rejects use of built-in names as func names
        err = reject_any("def print(x: int) -> unit = ()\nprint(1)")
        assert "print" in str(err) or "built-in" in str(err).lower()

    def test_funcdef_value_binding_type_is_function_type(self) -> None:
        r = accept_type('def f(x: int) -> text = "hi"\nf')
        f_ref = r.resolved.program.body.items[1]
        assert isinstance(f_ref, VarRef)
        t = r.node_types[f_ref.node_id]
        assert isinstance(t, FunctionType)


# ---------------------------------------------------------------------------
# Lambda
# ---------------------------------------------------------------------------


class TestLambda:
    def test_lambda_with_return_type(self) -> None:
        r = accept_type("fn(x: int) -> int => x")
        lam = r.resolved.program.body.items[0]
        assert isinstance(lam, Lambda)
        t = r.node_types[lam.node_id]
        assert t == FunctionType(params=(IntType(),), result=IntType())

    def test_lambda_without_return_type_inferred(self) -> None:
        r = accept_type("fn(x: int) => x")
        lam = r.resolved.program.body.items[0]
        assert isinstance(lam, Lambda)
        t = r.node_types[lam.node_id]
        assert t == FunctionType(params=(IntType(),), result=IntType())

    def test_lambda_body_type_mismatch(self) -> None:
        err = reject_type("fn(x: int) -> text => x")
        assert "mismatch" in str(err).lower() or "expected" in str(err).lower()

    def test_lambda_with_raise_body_needs_annotation(self) -> None:
        err = reject_type('fn() => raise Abort(message = "x")')
        assert "infer" in str(err).lower() or "return" in str(err).lower()

    def test_lambda_with_raise_and_annotation(self) -> None:
        r = accept_type('fn() -> int => raise Abort(message = "x")')
        assert r.resolved.program is not None

    def test_lambda_value_call(self) -> None:
        r = accept_type("let f = fn(x: int) -> int => x\nf(42)")
        assert r.resolved.program is not None

    def test_lambda_value_call_arity_mismatch(self) -> None:
        err = reject_type("let f = fn(x: int) -> int => x\nf(1, 2)")
        assert "arity" in str(err).lower() or "argument" in str(err).lower()

    def test_lambda_value_call_non_function_callee(self) -> None:
        err = reject_type("let x = 42\nx(1)")
        assert "function" in str(err).lower() or "callee" in str(err).lower()

    def test_lambda_value_call_named_args_rejected(self) -> None:
        err = reject_type("let f = fn(x: int) -> int => x\nf(x = 42)")
        assert "named" in str(err).lower() or "declared" in str(err).lower()


# ---------------------------------------------------------------------------
# If expressions
# ---------------------------------------------------------------------------


class TestIf:
    def test_if_else_unifies(self) -> None:
        r = accept_type("if true => 1 | else => 2")
        if_node = r.resolved.program.body.items[0]
        assert isinstance(if_node, If)
        t = r.node_types[if_node.node_id]
        assert t == IntType()

    def test_if_without_else_yields_unit(self) -> None:
        r = accept_type("if true => ()")
        if_node = r.resolved.program.body.items[0]
        assert isinstance(if_node, If)
        t = r.node_types[if_node.node_id]
        assert t == UnitType()

    def test_if_condition_must_be_bool(self) -> None:
        err = reject_type("if 1 => 2 | else => 3")
        assert "bool" in str(err).lower() or "condition" in str(err).lower()

    def test_if_branches_incompatible_types(self) -> None:
        err = reject_type('if true => 1 | else => "hello"')
        assert "incompatible" in str(err).lower() or "branches" in str(err).lower()

    def test_if_int_decimal_widening(self) -> None:
        r = accept_type("if true => 1 | else => 2.5")
        if_node = r.resolved.program.body.items[0]
        assert isinstance(if_node, If)
        t = r.node_types[if_node.node_id]
        assert t == DecimalType()

    def test_if_multiple_branches(self) -> None:
        r = accept_type("if true => 1 | true => 2 | else => 3")
        assert r.resolved.program is not None

    def test_if_annotation_context(self) -> None:
        r = accept_type("let x: int = if true => 1 | else => 2\nx")
        assert r.resolved.program is not None


# ---------------------------------------------------------------------------
# Case expressions
# ---------------------------------------------------------------------------


class TestCase:
    def test_case_wildcard_branch(self) -> None:
        r = accept_type('let x = 1\ncase x of | _ => "got it"')
        assert r.resolved.program is not None

    def test_case_var_pattern(self) -> None:
        r = accept_type("let x = 1\ncase x of | n => n")
        assert r.resolved.program is not None

    def test_case_literal_pattern(self) -> None:
        r = accept_type('let x = 1\ncase x of | 1 => "one" | _ => "other"')
        assert r.resolved.program is not None

    def test_case_enum_constructor_pattern(self) -> None:
        r = accept_type(
            "enum Status\n  | Pass\n  | Fail\nlet s = Pass()\n"
            "case s of | Status.Pass => 1 | Status.Fail => 2"
        )
        assert r.resolved.program is not None

    def test_case_incompatible_branch_types(self) -> None:
        err = reject_type('let x = 1\ncase x of | 1 => "a" | _ => 2')
        assert "incompatible" in str(err).lower() or "branches" in str(err).lower()

    def test_case_non_exhaustive_enum_warns(self) -> None:
        r = accept_type(
            "enum Status\n  | Pass\n  | Fail\nlet s = Pass()\n"
            "case s of | Status.Pass => 1"
        )
        assert any("Non-exhaustive" in w.message for w in r.warnings)

    def test_case_exhaustive_enum_no_warn(self) -> None:
        r = accept_type(
            "enum Status\n  | Pass\n  | Fail\nlet s = Pass()\n"
            "case s of | Status.Pass => 1 | Status.Fail => 2"
        )
        assert not any("Non-exhaustive" in w.message for w in r.warnings)

    def test_case_wildcard_suppresses_exhaustiveness_warning(self) -> None:
        r = accept_type(
            "enum Status\n  | Pass\n  | Fail\nlet s = Pass()\n"
            "case s of | _ => 1"
        )
        assert not any("Non-exhaustive" in w.message for w in r.warnings)

    def test_case_literal_incompatible_with_scrutinee(self) -> None:
        err = reject_type('let x = "hello"\ncase x of | 42 => 1 | _ => 2')
        assert "incompatible" in str(err).lower() or "pattern" in str(err).lower()

    def test_case_constructor_on_non_enum_raises(self) -> None:
        err = reject_type("let x = 1\ncase x of | Abort() => 1 | _ => 2")
        assert "enum" in str(err).lower() or "constructor" in str(err).lower()

    def test_case_with_constructor_field_pattern(self) -> None:
        r = accept_type(
            "enum Result\n  | Ok(value: int)\n  | Err(msg: text)\n"
            "let res = Ok(value = 42)\n"
            "case res of | Result.Ok(value = v) => v | Result.Err(msg = m) => 0"
        )
        assert r.resolved.program is not None


# ---------------------------------------------------------------------------
# Do loop
# ---------------------------------------------------------------------------


class TestDo:
    def test_do_yields_unit(self) -> None:
        r = accept_type("var i = 0\ndo\n  i := i + 1\nuntil i > 5")
        do_node = r.resolved.program.body.items[1]
        assert isinstance(do_node, Do)
        t = r.node_types[do_node.node_id]
        assert t == UnitType()

    def test_do_condition_must_be_bool(self) -> None:
        err = reject_type("var i = 0\ndo\n  i := i + 1\nuntil i")
        assert "bool" in str(err).lower() or "condition" in str(err).lower()


# ---------------------------------------------------------------------------
# Try/catch
# ---------------------------------------------------------------------------


class TestTryCatch:
    def test_try_catch_wildcard(self) -> None:
        r = accept_type("try 1 catch _ => 2")
        assert r.resolved.program is not None

    def test_try_catch_specific_exc_type(self) -> None:
        r = accept_type("try 1 catch Exception as e => 0")
        assert r.resolved.program is not None

    def test_try_catch_binding_accesses_attempts(self) -> None:
        r = accept_type("try 0 catch AgentParseError as e => e.attempts")
        assert r.resolved.program is not None

    def test_try_catch_unknown_exc_type(self) -> None:
        err = reject_type("try 1 catch UnknownError => 0")
        assert "exception" in str(err).lower() or "unknown" in str(err).lower()

    def test_try_catch_incompatible_types(self) -> None:
        err = reject_type('try 1 catch _ => "hello"')
        assert "incompatible" in str(err).lower() or "branches" in str(err).lower()

    def test_try_unified_int_decimal(self) -> None:
        r = accept_type("try 1 catch _ => 2.5")
        try_node = r.resolved.program.body.items[0]
        assert isinstance(try_node, Try)
        t = r.node_types[try_node.node_id]
        assert t == DecimalType()

    def test_try_wildcard_binding_accesses_message(self) -> None:
        # body is text so handler text=text is compatible
        r = accept_type('try "hello" catch _ as e => e.message')
        assert r.resolved.program is not None

    def test_try_catch_abort_binding(self) -> None:
        r = accept_type('try "x" catch Abort as e => e.message')
        assert r.resolved.program is not None


# ---------------------------------------------------------------------------
# Raise
# ---------------------------------------------------------------------------


class TestRaise:
    def test_raise_abort_ok(self) -> None:
        r = accept_type('raise Abort(message = "error")')
        raise_node = r.resolved.program.body.items[0]
        assert isinstance(raise_node, Raise)
        t = r.node_types[raise_node.node_id]
        assert isinstance(t, BottomType)

    def test_raise_non_exception_rejected(self) -> None:
        err = reject_type("raise 42")
        assert "exception" in str(err).lower() or "raise" in str(err).lower()

    def test_raise_bottom_assignable_to_any(self) -> None:
        # raise can be used where any type is expected (annotated binding)
        r = accept_type('let x: int = raise Abort(message = "err")\nx')
        assert r.resolved.program is not None

    def test_raise_in_funcdef_body(self) -> None:
        r = accept_type('def f() -> text = raise Abort(message = "err")\nf()')
        assert r.resolved.program is not None


# ---------------------------------------------------------------------------
# Template interpolation
# ---------------------------------------------------------------------------


class TestTemplate:
    def test_plain_template_is_text(self) -> None:
        r = accept_type('let s = "hello world"\ns')
        assert r.resolved.program is not None

    def test_interpolated_template(self) -> None:
        r = accept_type('let x = 42\nlet s = "${x}"\ns')
        assert r.resolved.program is not None

    def test_interpolated_function_accepted(self) -> None:
        accept_type('let f = fn(x: int) -> int => x\n"${f}"')

    def test_interpolated_agent_accepted(self) -> None:
        accept_type('agent a\n"${a}"')

    def test_interpolated_int_is_ok(self) -> None:
        r = accept_type('let n = 1\n"n is ${n}"')
        assert r.resolved.program is not None

    def test_interpolated_null_is_ok(self) -> None:
        r = accept_type('"${null}"')
        assert r.resolved.program is not None

    def test_interpolated_list_literal(self) -> None:
        r = accept_type('"${[1, 2, 3]}"')
        assert r.resolved.program is not None

    def test_interpolated_dict_literal(self) -> None:
        # dict in template uses double-brace: ${{ key: val }}
        r = accept_type('"${{"a": 1}}"')
        assert r.resolved.program is not None

    def test_interpolated_nested_json(self) -> None:
        # nested dict inside template interpolation
        r = accept_type('let n = 42\n"${{"a": n}}"')
        assert r.resolved.program is not None

    def test_interpolated_non_json_in_dict_rejected(self) -> None:
        err = reject_type('record R\n  x: int\nlet r = R(x = 1)\n"${{"a": r}}"')
        assert "json" in str(err).lower() or "mismatch" in str(err).lower()


# ---------------------------------------------------------------------------
# Binary operators
# ---------------------------------------------------------------------------


class TestBinaryOps:
    def test_add_int(self) -> None:
        r = accept_type("1 + 2")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == IntType()

    def test_add_decimal(self) -> None:
        r = accept_type("1.5 + 2.5")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == DecimalType()

    def test_add_int_decimal_widens(self) -> None:
        r = accept_type("1 + 2.5")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == DecimalType()

    def test_add_text(self) -> None:
        r = accept_type('"a" + "b"')
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == TextType()

    def test_add_type_mismatch(self) -> None:
        err = reject_type('1 + "hello"')
        assert "+" in str(err) or "numeric" in str(err).lower()

    def test_sub(self) -> None:
        r = accept_type("5 - 3")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == IntType()

    def test_mul(self) -> None:
        r = accept_type("2 * 3")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == IntType()

    def test_div_yields_decimal(self) -> None:
        r = accept_type("5 / 2")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == DecimalType()

    def test_eq_same_type(self) -> None:
        r = accept_type("1 == 2")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == BoolType()

    def test_neq_same_type(self) -> None:
        r = accept_type('"a" != "b"')
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == BoolType()

    def test_lt(self) -> None:
        r = accept_type("1 < 2")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == BoolType()

    def test_gt(self) -> None:
        r = accept_type("2 > 1")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == BoolType()

    def test_le(self) -> None:
        r = accept_type("1 <= 2")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == BoolType()

    def test_ge(self) -> None:
        r = accept_type("2 >= 1")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == BoolType()

    def test_and(self) -> None:
        r = accept_type("true and false")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == BoolType()

    def test_or(self) -> None:
        r = accept_type("true or false")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == BoolType()

    def test_in_text(self) -> None:
        r = accept_type('"a" in "abc"')
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == BoolType()

    def test_in_list(self) -> None:
        r = accept_type("let xs: list[int] = [1, 2, 3]\n1 in xs")
        node = r.resolved.program.body.items[1]
        assert r.node_types[node.node_id] == BoolType()

    def test_in_dict(self) -> None:
        r = accept_type('let d: dict[text, int] = {"a": 1}\n"a" in d')
        node = r.resolved.program.body.items[1]
        assert r.node_types[node.node_id] == BoolType()

    def test_and_non_bool_left_raises(self) -> None:
        err = reject_type("1 and true")
        assert "and" in str(err).lower() or "bool" in str(err).lower()

    def test_and_non_bool_right_raises(self) -> None:
        err = reject_type("true and 1")
        assert "and" in str(err).lower() or "bool" in str(err).lower()

    def test_or_non_bool_left_raises(self) -> None:
        err = reject_type("1 or true")
        assert "or" in str(err).lower() or "bool" in str(err).lower()

    def test_eq_different_types_raises(self) -> None:
        err = reject_type('1 == "hello"')
        assert "same" in str(err).lower() or "equality" in str(err).lower()

    # Transitive no-equality: function/agent/unit inside containers/records/enums.
    def test_eq_list_of_fn_raises(self) -> None:
        err = reject_type(
            "def f(n: int) -> int = n\n"
            "def g(n: int) -> int = n\n"
            "let fs: list[(int) -> int] = [f, g]\n"
            "let gs: list[(int) -> int] = [f]\n"
            "let r = (fs == gs)\nr"
        )
        assert "equality" in str(err).lower()

    def test_eq_dict_of_fn_raises(self) -> None:
        err = reject_type(
            "def f(n: int) -> int = n\n"
            'let d1: dict[text, (int) -> int] = {"a": f}\n'
            'let d2: dict[text, (int) -> int] = {"b": f}\n'
            "let r = (d1 == d2)\nr"
        )
        assert "equality" in str(err).lower()

    def test_eq_record_with_fn_field_raises(self) -> None:
        err = reject_type(
            "def f(n: int) -> int = n\n"
            "record R\n  cb: (int) -> int\n"
            "let r1 = R(cb = f)\n"
            "let r2 = R(cb = f)\n"
            "let result = (r1 == r2)\nresult"
        )
        assert "equality" in str(err).lower()

    def test_eq_enum_with_fn_field_raises(self) -> None:
        err = reject_type(
            "def f(n: int) -> int = n\n"
            "enum E\n  | A(cb: (int) -> int)\n  | B\n"
            "let e1 = E.A(cb = f)\n"
            "let e2 = E.B\n"
            "let result = (e1 == e2)\nresult"
        )
        assert "equality" in str(err).lower()

    def test_eq_list_of_agent_raises(self) -> None:
        err = reject_type(
            "agent reviewer\n"
            "let as1: list[agent] = [reviewer]\n"
            "let as2: list[agent] = [reviewer]\n"
            "let r = (as1 == as2)\nr"
        )
        assert "equality" in str(err).lower()

    def test_eq_list_of_unit_raises(self) -> None:
        err = reject_type(
            "let us1: list[unit] = [()]\n"
            "let us2: list[unit] = [()]\n"
            "let r = (us1 == us2)\nr"
        )
        assert "equality" in str(err).lower()

    def test_eq_record_with_agent_field_raises(self) -> None:
        err = reject_type(
            "agent reviewer\n"
            "record Task\n  name: text\n  assignee: agent\n"
            "let t1 = Task(name = \"a\", assignee = reviewer)\n"
            "let t2 = Task(name = \"b\", assignee = reviewer)\n"
            "let result = (t1 == t2)\nresult"
        )
        assert "equality" in str(err).lower()

    # Regression: bare function/agent/unit still rejected.
    def test_eq_bare_fn_raises(self) -> None:
        err = reject_type(
            "def f(n: int) -> int = n\n"
            "let r = (f == f)\nr"
        )
        assert "equality" in str(err).lower()

    def test_eq_bare_agent_raises(self) -> None:
        err = reject_type(
            "agent reviewer\n"
            "let r = (reviewer == reviewer)\nr"
        )
        assert "equality" in str(err).lower()

    def test_eq_bare_unit_raises(self) -> None:
        err = reject_type("let r = (() == ())\nr")
        assert "equality" in str(err).lower()

    # Still-accept: lists/records/dicts of equatable scalars must stay green.
    def test_eq_list_of_int_accepted(self) -> None:
        accept_type("let xs: list[int] = [1, 2]\nlet ys: list[int] = [3]\nlet r = (xs == ys)\nr")

    def test_eq_dict_of_text_accepted(self) -> None:
        accept_type(
            'let d1: dict[text, text] = {"a": "x"}\n'
            'let d2: dict[text, text] = {"b": "y"}\n'
            "let r = (d1 == d2)\nr"
        )

    def test_eq_json_accepted(self) -> None:
        accept_type("let a: json = null\nlet b: json = null\nlet r = (a == b)\nr")

    def test_eq_int_decimal_cross_accepted(self) -> None:
        accept_type("let r = (1 == 1.0)\nr")

    def test_eq_record_with_scalar_fields_accepted(self) -> None:
        accept_type(
            "record Point\n  x: int\n  y: int\n"
            "let p1 = Point(x = 0, y = 0)\n"
            "let p2 = Point(x = 1, y = 1)\n"
            "let r = (p1 == p2)\nr"
        )

    def test_eq_enum_with_scalar_fields_accepted(self) -> None:
        accept_type(
            "enum Color\n  | Red\n  | Blue\n"
            "let r = (Color.Red == Color.Blue)\nr"
        )

    def test_ordering_non_numeric_non_text_raises(self) -> None:
        err = reject_type("true < false")
        assert "ordering" in str(err).lower() or "numeric" in str(err).lower()

    def test_in_type_mismatch_raises(self) -> None:
        err = reject_type('let xs: list[int] = [1, 2]\n"hello" in xs')
        assert "in" in str(err).lower() or "mismatch" in str(err).lower()

    def test_in_invalid_container_raises(self) -> None:
        err = reject_type("1 in 2")
        assert "in" in str(err).lower()

    def test_div_non_numeric_raises(self) -> None:
        err = reject_type('"a" / "b"')
        assert "/" in str(err) or "numeric" in str(err).lower()

    def test_sub_non_numeric_raises(self) -> None:
        err = reject_type('"a" - "b"')
        assert "-" in str(err) or "numeric" in str(err).lower()

    def test_mul_non_numeric_raises(self) -> None:
        err = reject_type('"a" * "b"')
        assert "*" in str(err) or "numeric" in str(err).lower()


# ---------------------------------------------------------------------------
# Unary operators
# ---------------------------------------------------------------------------


class TestUnaryOps:
    def test_not_bool(self) -> None:
        r = accept_type("not true")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == BoolType()

    def test_not_non_bool_raises(self) -> None:
        err = reject_type("not 1")
        assert "not" in str(err).lower() or "bool" in str(err).lower()

    def test_neg_int(self) -> None:
        r = accept_type("-5")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == IntType()

    def test_neg_decimal(self) -> None:
        r = accept_type("-3.14")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == DecimalType()

    def test_neg_non_numeric_raises(self) -> None:
        err = reject_type('-"hello"')
        assert "numeric" in str(err).lower() or "'-'" in str(err)


# ---------------------------------------------------------------------------
# Field access
# ---------------------------------------------------------------------------


class TestFieldAccess:
    def test_record_field_access(self) -> None:
        r = accept_type("record Point\n  x: int\n  y: int\nlet p = Point(x = 1, y = 2)\np.x")
        assert r.resolved.program is not None

    def test_record_unknown_field_raises(self) -> None:
        err = reject_type("record Point\n  x: int\nlet p = Point(x = 1)\np.z")
        assert "field" in str(err).lower() or "z" in str(err)

    def test_exception_field_access(self) -> None:
        # catch Abort and access its message field
        r = accept_type('try "x" catch Abort as e => e.message')
        assert r.resolved.program is not None

    def test_exception_unknown_field_raises(self) -> None:
        err = reject_type("try 1 catch Abort as e => e.nonexistent")
        assert "field" in str(err).lower() or "nonexistent" in str(err)

    def test_field_access_non_record_raises(self) -> None:
        err = reject_type("let x = 42\nx.field")
        assert "record" in str(err).lower() or "field" in str(err).lower()


# ---------------------------------------------------------------------------
# Is test
# ---------------------------------------------------------------------------


class TestIsTest:
    def test_is_enum_variant(self) -> None:
        r = accept_type(
            "enum Status\n  | Pass\n  | Fail\nlet s = Pass()\ns is Status.Pass"
        )
        node = r.resolved.program.body.items[2]
        assert r.node_types[node.node_id] == BoolType()

    def test_is_not(self) -> None:
        r = accept_type(
            "enum Status\n  | Pass\n  | Fail\nlet s = Pass()\ns is not Status.Pass"
        )
        assert r.resolved.program is not None

    def test_is_non_enum_raises(self) -> None:
        err = reject_type("let x = 1\nx is Status")
        assert "enum" in str(err).lower()

    def test_is_unknown_variant_raises(self) -> None:
        err = reject_type(
            "enum Status\n  | Pass\n  | Fail\nlet s = Pass()\ns is Status.Gone"
        )
        assert "variant" in str(err).lower()

    def test_is_test_wrong_qualifier_raises(self) -> None:
        err = reject_type(
            "enum A\n  | X\nenum B\n  | X\nlet a = A.X()\na is B.X"
        )
        assert "qualifier" in str(err).lower() or "enum" in str(err).lower()


# ---------------------------------------------------------------------------
# Constructor expressions
# ---------------------------------------------------------------------------


class TestConstructors:
    def test_record_constructor(self) -> None:
        r = accept_type("record Point\n  x: int\n  y: int\nPoint(x = 1, y = 2)")
        assert r.resolved.program is not None

    def test_record_missing_field_raises(self) -> None:
        err = reject_type("record Point\n  x: int\n  y: int\nPoint(x = 1)")
        assert "missing" in str(err).lower() or "field" in str(err).lower()

    def test_record_unknown_field_raises(self) -> None:
        err = reject_type("record Point\n  x: int\nPoint(x = 1, z = 2)")
        assert (
            "no field" in str(err).lower()
            or "field" in str(err).lower()
            or "unknown" in str(err).lower()
        )

    def test_record_duplicate_arg_raises(self) -> None:
        # Parser catches duplicate field args
        err = reject_any("record Point\n  x: int\nPoint(x = 1, x = 2)")
        assert "duplicate" in str(err).lower() or "x" in str(err)

    def test_record_field_type_mismatch(self) -> None:
        err = reject_type('record Point\n  x: int\nPoint(x = "hello")')
        assert "mismatch" in str(err).lower() or "expected" in str(err).lower()

    def test_enum_variant_qualified(self) -> None:
        r = accept_type("enum Status\n  | Pass\n  | Fail\nStatus.Pass()")
        assert r.resolved.program is not None

    def test_enum_variant_unqualified_unique(self) -> None:
        r = accept_type("enum Status\n  | Pass\n  | Fail\nPass()")
        assert r.resolved.program is not None

    def test_enum_variant_ambiguous_raises(self) -> None:
        # Ambiguity is now detected at scope-resolution time (AglScopeError).
        err = reject_any("enum A\n  | Pass\nenum B\n  | Pass\nPass()")
        assert "ambiguous" in str(err).lower()

    def test_enum_variant_unknown_raises(self) -> None:
        err = reject_type("enum Status\n  | Pass\nStatus.Gone()")
        assert "variant" in str(err).lower()

    def test_exception_constructor(self) -> None:
        r = accept_type('Abort(message = "error")')
        assert r.resolved.program is not None

    def test_abstract_exception_not_constructible(self) -> None:
        err = reject_type('Exception(message = "e")')
        assert "abstract" in str(err).lower() or "constructible" in str(err).lower()

    def test_unknown_constructor_raises(self) -> None:
        # Unknown names are now caught at scope-resolution time (AglScopeError).
        err = reject_any("Unknown(x = 1)")
        assert "unknown" in str(err).lower() or "Unknown" in str(err) or "not defined" in str(err)

    def test_enum_variant_with_fields(self) -> None:
        # enum variants can have named fields
        r = accept_type(
            "enum Result\n  | Ok(value: int)\n  | Err(msg: text)\nOk(value = 42)"
        )
        assert r.resolved.program is not None

    def test_qualified_constructor_wrong_enum_raises(self) -> None:
        err = reject_type("enum A\n  | X\nenum B\n  | Y\nA.Y()")
        assert "variant" in str(err).lower()

    def test_qualified_constructor_not_enum_raises(self) -> None:
        err = reject_type("record R\n  x: int\nR.Something()")
        assert "enum" in str(err).lower()


# ---------------------------------------------------------------------------
# Constructor ref dispatch (VarRef/Call/FieldAccess paths)
# ---------------------------------------------------------------------------


class TestConstructorRefDispatch:
    """Verify construction via the new VarRef/Call/FieldAccess constructor paths."""

    def test_bare_varref_nullary_variant(self) -> None:
        # Bare nullary variant as VarRef → zero-arg construction
        r = accept_type("enum Status\n  | Pass\n  | Fail\nlet s = Pass()\ns")
        assert r.resolved.program is not None

    def test_call_varref_record_constructor(self) -> None:
        # Record construction via Call(callee=VarRef)
        r = accept_type("record Box\n  value: int\nBox(value = 1)")
        assert r.resolved.program is not None

    def test_call_varref_enum_payload_variant(self) -> None:
        # Payload variant via Call(callee=VarRef)
        r = accept_type("enum Option\n  | none\n  | some(value: int)\nsome(value = 1)")
        assert r.resolved.program is not None

    def test_qualified_call_enum_variant(self) -> None:
        # Qualified construction: Option.some(value = 1)
        r = accept_type("enum Option\n  | none\n  | some(value: int)\nOption.some(value = 1)")
        assert r.resolved.program is not None

    def test_qualified_bare_nullary_variant(self) -> None:
        # Bare qualified constructor: FieldAccess → zero-arg construction
        r = accept_type("enum Status\n  | Pass\n  | Fail\nStatus.Pass()")
        assert r.resolved.program is not None

    def test_missing_field_still_errors(self) -> None:
        err = reject_type("record Box\n  value: int\nBox()")
        assert "missing" in str(err).lower() or "field" in str(err).lower()

    def test_unknown_field_still_errors(self) -> None:
        err = reject_type("record Box\n  value: int\nBox(value = 1, extra = 2)")
        assert (
            "no field" in str(err).lower()
            or "field" in str(err).lower()
            or "unknown" in str(err).lower()
        )

    def test_field_type_mismatch_still_errors(self) -> None:
        err = reject_type('record Box\n  value: int\nBox(value = "hello")')
        assert "mismatch" in str(err).lower() or "expected" in str(err).lower()

    def test_qualified_variant_not_found_errors(self) -> None:
        err = reject_type("enum Status\n  | Pass\n  | Fail\nStatus.Missing()")
        assert "variant" in str(err).lower()

    def test_qualified_non_enum_errors(self) -> None:
        err = reject_type("record R\n  x: int\nR.Something()")
        assert "enum" in str(err).lower()

    def test_exception_constructor_via_new_dispatch(self) -> None:
        # Exception constructors go through the new unqualified path
        r = accept_type('Abort(message = "error")')
        assert r.resolved.program is not None

    def test_abstract_exception_rejected_in_new_dispatch(self) -> None:
        err = reject_type('Exception(message = "e")')
        assert "abstract" in str(err).lower() or "constructible" in str(err).lower()

    def test_positional_arg_on_unqualified_constructor_rejected(self) -> None:
        # Constructors only accept named args; positional arg must be rejected.
        err = reject_type("enum E\n  | Pass\nPass(1)")
        assert "named" in str(err).lower() or "positional" in str(err).lower()

    def test_type_arg_on_unqualified_constructor_rejected(self) -> None:
        # Type arguments on constructors are not yet supported.
        err = reject_type("enum E\n  | Pass\nPass::[int]()")
        assert "type argument" in str(err).lower() or "not supported" in str(err).lower()

    def test_positional_arg_on_qualified_constructor_rejected(self) -> None:
        # Qualified constructor with positional arg is rejected.
        err = reject_type("enum E\n  | Pass\nE.Pass(1)")
        assert "named" in str(err).lower() or "positional" in str(err).lower()

    def test_type_arg_on_qualified_constructor_rejected(self) -> None:
        # Type arguments on qualified constructors are not yet supported.
        # The grammar allows ::[ on a name but not on a qualified field access,
        # so this is caught as a type checker error when using the VarRef form.
        err = reject_type("enum E\n  | Pass\nPass::[int]()")
        assert "type argument" in str(err).lower() or "not supported" in str(err).lower()

    def test_single_field_variant_positional(self) -> None:
        # Single-field enum variant with no markers → STANDARD zone → positional works.
        r = accept_type("enum Opt\n  | Some(value: int)\n  | None\nSome(42)")
        assert r.resolved.program is not None

    def test_single_field_variant_named_still_works(self) -> None:
        # Single-field variant: named arg still works even though zone is STANDARD.
        r = accept_type("enum Opt\n  | Some(value: int)\nSome(value = 5)")
        assert r.resolved.program is not None

    def test_std_record_positional(self) -> None:
        # Record with @std marker → positional args allowed.
        r = accept_type("record P\n  @std\n  x: int\n  y: int\nP(1, 2)")
        assert r.resolved.program is not None

    def test_std_record_mixed_pos_named(self) -> None:
        # Record with @std: positional then named allowed.
        r = accept_type("record P\n  @std\n  x: int\n  y: int\nP(1, y = 2)")
        assert r.resolved.program is not None

    def test_pos_only_field_by_name_rejected(self) -> None:
        # A pos-only field cannot be passed by name.
        err = reject_type("record R\n  @pos\n  x: int\n  y: int\nR(x = 1, y = 2)")
        assert "positional-only" in str(err).lower() or "positional" in str(err).lower()

    def test_multi_field_variant_nonbare_positional_rejected(self) -> None:
        # Non-bare positional in named-only multi-field variant → error.
        err = reject_type("enum E\n  | F(x: int, y: int)\nF(1, 2)")
        assert "named-only" in str(err).lower() or "positional" in str(err).lower()

    def test_get_constructor_field_kinds_no_graph_table(self) -> None:
        # get_constructor_field_kinds in single-program mode: no graph table → None for unknown.
        from agm.agl.modules.ids import ModuleId
        env = TypeEnvironment()
        lib_id = ModuleId.from_dotted("lib")
        result = env.get_constructor_field_kinds("Unknown", None, module_id=lib_id)
        assert result is None


# ---------------------------------------------------------------------------
# Bare constructors as function values with explicit type arguments
# (`Some::[int]` ≡ `fn (x: int) => Some(x)`, `None::[int]` constructs directly)
# ---------------------------------------------------------------------------


class TestBareConstructorTypeApply:
    """Bare constructor references accept explicit type arguments in value
    position, producing a function value (payload variant) or the constructed
    nominal value (nullary variant)."""

    _OPT = "enum Option[T]\n  | none\n  | some(value: T)\n"

    def test_bare_payload_constructor_type_apply_is_function_value(self) -> None:
        r = accept_type(self._OPT + "let f = some::[int]\nf")
        prog = r.resolved.program
        f_ref = prog.body.items[-1]
        assert isinstance(f_ref, VarRef)
        assert r.node_types[f_ref.node_id] == FunctionType(
            params=(IntType(),), result=EnumType("Option", {}, type_args=(IntType(),))
        )

    def test_bare_nullary_constructor_type_apply_is_constructed_value(self) -> None:
        r = accept_type(self._OPT + "let z = none::[int]\nz")
        prog = r.resolved.program
        z_ref = prog.body.items[-1]
        assert isinstance(z_ref, VarRef)
        assert r.node_types[z_ref.node_id] == EnumType(
            "Option", {}, type_args=(IntType(),)
        )

    def test_bare_payload_constructor_callable(self) -> None:
        # `some::[int]` applied positionally yields the constructed enum value.
        r = accept_type(self._OPT + "let v = (some::[int])(7)\nv")
        prog = r.resolved.program
        v_ref = prog.body.items[-1]
        assert isinstance(v_ref, VarRef)
        assert r.node_types[v_ref.node_id] == EnumType(
            "Option", {"some": {"value": IntType()}, "none": {}}, type_args=(IntType(),)
        )

    def test_bare_nullary_constructor_used_in_context(self) -> None:
        r = accept_type(self._OPT + "let z: Option[int] = none::[int]\nz")
        assert r.resolved.program is not None

    def test_qualified_payload_constructor_type_apply(self) -> None:
        r = accept_type(self._OPT + "let f = Option.some::[int]\nf")
        prog = r.resolved.program
        f_ref = prog.body.items[-1]
        assert isinstance(f_ref, VarRef)
        assert r.node_types[f_ref.node_id] == FunctionType(
            params=(IntType(),), result=EnumType("Option", {}, type_args=(IntType(),))
        )

    def test_qualified_nullary_constructor_type_apply(self) -> None:
        r = accept_type(self._OPT + "let z = Option.none::[int]\nz")
        prog = r.resolved.program
        z_ref = prog.body.items[-1]
        assert isinstance(z_ref, VarRef)
        assert r.node_types[z_ref.node_id] == EnumType(
            "Option", {}, type_args=(IntType(),)
        )

    def test_qualified_payload_constructor_callable(self) -> None:
        r = accept_type(self._OPT + "let v = (Option.some::[int])(7)\nv")
        assert r.resolved.program is not None

    def test_non_generic_constructor_type_apply_rejected(self) -> None:
        err = reject_type("enum E\n  | Pass\nlet f = Pass::[int]\nf")
        assert "not a generic constructor" in str(err).lower()

    def test_qualified_non_generic_constructor_type_apply_rejected(self) -> None:
        err = reject_type("enum E\n  | Pass\nlet f = E.Pass::[int]\nf")
        assert "not a generic constructor" in str(err).lower()

    def test_wrong_arity_type_apply_rejected(self) -> None:
        err = reject_type(self._OPT + "let f = some::[int, text]\nf")
        assert "type argument" in str(err).lower()

    def test_qualified_wrong_arity_type_apply_rejected(self) -> None:
        err = reject_type(self._OPT + "let f = Option.some::[int, text]\nf")
        assert "type argument" in str(err).lower()

    def test_generic_record_constructor_type_apply(self) -> None:
        # A generic record constructor as a value with explicit type args.
        r = accept_type(
            "record Box[T]\n  value: T\nlet f = Box::[int]\nlet v = f(9)\nf"
        )
        prog = r.resolved.program
        f_ref = prog.body.items[-1]
        assert isinstance(f_ref, VarRef)
        assert r.node_types[f_ref.node_id] == FunctionType(
            params=(IntType(),), result=RecordType("Box", {}, type_args=(IntType(),))
        )


# ---------------------------------------------------------------------------
# List literals
# ---------------------------------------------------------------------------


class TestListLiterals:
    def test_list_int(self) -> None:
        r = accept_type("[1, 2, 3]")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == ListType(elem=IntType())

    def test_list_text(self) -> None:
        r = accept_type('["a", "b"]')
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == ListType(elem=TextType())

    def test_list_empty_with_annotation(self) -> None:
        r = accept_type("let xs: list[int] = []\nxs")
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        assert r.type_env.get_binding_type(decl.node_id) == ListType(elem=IntType())

    def test_list_empty_no_annotation_raises(self) -> None:
        err = reject_type("[]")
        assert "annotation" in str(err).lower() or "empty" in str(err).lower()

    def test_list_inconsistent_elements_raises(self) -> None:
        err = reject_type('["a", 1]')
        assert "inconsistent" in str(err).lower() or "type" in str(err).lower()

    def test_list_int_decimal_widening(self) -> None:
        r = accept_type("[1, 2.5]")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == ListType(elem=DecimalType())

    def test_list_in_json_context(self) -> None:
        r = accept_type("let xs: json = [1, 2, 3]\nxs")
        assert r.resolved.program is not None

    def test_list_record_in_json_raises(self) -> None:
        err = reject_type("record R\n  x: int\nlet xs: list[json] = [R(x = 1)]\nxs")
        assert "json" in str(err).lower() or "mismatch" in str(err).lower()


# ---------------------------------------------------------------------------
# Dict literals
# ---------------------------------------------------------------------------


class TestDictLiterals:
    def test_dict_text_int(self) -> None:
        r = accept_type('{"a": 1, "b": 2}')
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == DictType(value=IntType())

    def test_dict_empty_with_annotation(self) -> None:
        r = accept_type('let d: dict[text, int] = {}\nd')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        assert r.type_env.get_binding_type(decl.node_id) == DictType(value=IntType())

    def test_dict_empty_no_annotation_raises(self) -> None:
        err = reject_type("{}")
        assert "annotation" in str(err).lower() or "empty" in str(err).lower()

    def test_dict_duplicate_key_raises(self) -> None:
        err = reject_type('{"a": 1, "a": 2}')
        assert "duplicate" in str(err).lower() or "key" in str(err).lower()

    def test_dict_inconsistent_values_raises(self) -> None:
        err = reject_type('{"a": 1, "b": "text"}')
        assert "inconsistent" in str(err).lower() or "type" in str(err).lower()

    def test_dict_in_json_context(self) -> None:
        r = accept_type('let d: json = {"a": 1}\nd')
        assert r.resolved.program is not None


# ---------------------------------------------------------------------------
# Type declarations: record/enum/alias
# ---------------------------------------------------------------------------


class TestTypeDeclarations:
    def test_record_def(self) -> None:
        r = accept_type("record Point\n  x: int\n  y: int\nPoint(x = 1, y = 2)")
        assert r.resolved.program is not None

    def test_enum_def(self) -> None:
        r = accept_type("enum Color\n  | Red\n  | Blue\nRed()")
        assert r.resolved.program is not None

    def test_type_alias(self) -> None:
        r = accept_type("type MyInt = int\nlet x: MyInt = 42\nx")
        assert r.resolved.program is not None

    def test_parameterized_alias_param_in_body(self) -> None:
        # A parameterized alias body may reference its own type parameter, and
        # applying it substitutes the argument (regression: the type-builder
        # previously registered aliases with 0 params and dropped body type vars).
        r = accept_type("type Wrap[A] = list[A]\nlet w: Wrap[int] = [1, 2]\nw")
        assert r.resolved.program is not None

    def test_parameterized_alias_unused_params(self) -> None:
        r = accept_type(
            "type Pair[A, B] = dict[text, json]\n"
            'let p: Pair[int, text] = {a: 1}\np'
        )
        assert r.resolved.program is not None

    def test_parameterized_alias_arity_mismatch_rejected(self) -> None:
        err = reject_type("type Wrap[A] = list[A]\nlet w: Wrap[int, text] = [1]\nw")
        assert "type argument" in str(err).lower()

    def test_parameterized_alias_substitution_enforced(self) -> None:
        # Wrap[int] resolves to list[int]; a text element is rejected.
        err = reject_type('type Wrap[A] = list[A]\nlet w: Wrap[int] = ["x"]\nw')
        assert (
            "int" in str(err).lower()
            or "text" in str(err).lower()
            or "mismatch" in str(err).lower()
        )

    def test_parameterized_alias_mutual_cycle_rejected(self) -> None:
        # A parameterized-alias cycle is a clean diagnostic, not a raw recursion
        # crash (the applied-type resolution path needs the same cycle guard as
        # the bare-name path).
        err = reject_type("type A[X] = B[X]\ntype B[Y] = A[Y]\nlet z: A[int] = 1\nz")
        assert "cycle" in str(err).lower()

    def test_parameterized_alias_self_cycle_rejected(self) -> None:
        err = reject_type("type A[X] = A[X]\nlet z: A[int] = 1\nz")
        assert "cycle" in str(err).lower()

    def test_duplicate_type_name_raises(self) -> None:
        err = reject_any("record A\n  x: int\nrecord A\n  y: int\nA(x = 1)")
        assert "already declared" in str(err).lower() or "duplicate" in str(err).lower()

    def test_type_builder_duplicate_type_name_guard_raises(self) -> None:
        program = parse_program("record A\n  x: int\nrecord A\n  y: int\nA(x = 1)")
        with pytest.raises(AglTypeError) as exc_info:
            _TypeBuilder(TypeEnvironment()).collect(program)
        assert "already declared" in str(exc_info.value).lower()

    def test_record_recursive_raises(self) -> None:
        err = reject_type("record Node\n  child: Node\nNode(child = Node(child = ()))")
        assert "recursive" in str(err).lower()

    def test_enum_recursive_raises(self) -> None:
        err = reject_type(
            "enum List\n  | Cons(value: int, rest: List)\n  | Nil\nNil()"
        )
        assert "recursive" in str(err).lower()

    def test_record_duplicate_field_raises(self) -> None:
        err = reject_type("record R\n  x: int\n  x: text\nR(x = 1)")
        assert "duplicate" in str(err).lower() or "field" in str(err).lower()

    def test_enum_duplicate_variant_raises(self) -> None:
        err = reject_type("enum E\n  | A\n  | A\nA()")
        assert "duplicate" in str(err).lower() or "variant" in str(err).lower()

    def test_alias_cycle_raises(self) -> None:
        err = reject_type("type A = B\ntype B = A\n1")
        assert "cycle" in str(err).lower()

    def test_record_not_json_shaped(self) -> None:
        err = reject_type("record R\n  x: int\nlet r: json = R(x = 1)\nr")
        assert "json" in str(err).lower() or "mismatch" in str(err).lower()

    def test_enum_not_json_shaped(self) -> None:
        err = reject_type("enum E\n  | A\nlet e: json = A()\ne")
        assert "json" in str(err).lower() or "mismatch" in str(err).lower()


# ---------------------------------------------------------------------------
# Var/assignment
# ---------------------------------------------------------------------------


class TestVarAssign:
    def test_assign_updates_var(self) -> None:
        r = accept_type("var x: int = 0\nx := 42\nx")
        assert r.resolved.program is not None

    def test_assign_type_mismatch_raises(self) -> None:
        err = reject_type('var x: int = 0\nx := "hello"')
        assert "mismatch" in str(err).lower() or "expected" in str(err).lower()

    def test_var_bottom_inference_raises(self) -> None:
        err = reject_type('var x = raise Abort(message = "e")\nx')
        assert "infer" in str(err).lower() or "raises" in str(err).lower()

    def test_let_bottom_inference_raises(self) -> None:
        err = reject_type('let x = raise Abort(message = "e")\nx')
        assert "infer" in str(err).lower() or "raises" in str(err).lower()

    def test_var_with_annotation_allows_bottom(self) -> None:
        r = accept_type('var x: int = raise Abort(message = "e")\nx')
        assert r.resolved.program is not None


# ---------------------------------------------------------------------------
# ParsePolicy constructors
# ---------------------------------------------------------------------------


class TestParsePolicy:
    def test_on_parse_error_abort(self) -> None:
        r = accept_type('let n: int = ask("Q", on_parse_error = Abort())\nn')
        assert r.call_sites[0].parse_policy == "abort"

    def test_on_parse_error_retry(self) -> None:
        r = accept_type('let n: int = ask("Q", on_parse_error = Retry(n = 5))\nn')
        assert r.call_sites[0].parse_policy == "retry[5]"

    def test_on_parse_error_invalid_constructor_raises(self) -> None:
        err = reject_type('let n: int = ask("Q", on_parse_error = 42)\nn')
        assert "on_parse_error" in str(err).lower() or "ParsePolicy" in str(err)

    def test_on_parse_error_abort_with_extra_args_raises(self) -> None:
        err = reject_type('let n: int = ask("Q", on_parse_error = Abort(message = "x"))\nn')
        assert "on_parse_error" in str(err).lower() or "Abort" in str(err)

    def test_on_parse_error_retry_no_n_raises(self) -> None:
        err = reject_type('let n: int = ask("Q", on_parse_error = Retry())\nn')
        assert "on_parse_error" in str(err).lower() or "Retry" in str(err)

    def test_on_parse_error_wrong_qualifier_raises(self) -> None:
        # 'Other' is not a declared type name, so this fails at scope time.
        err = reject_any('let n: int = ask("Q", on_parse_error = Other.Abort())\nn')
        err_str = str(err).lower()
        assert "on_parse_error" in err_str or "ParsePolicy" in str(err) or "Other" in str(err)

    def test_on_parse_error_text_target_warns(self) -> None:
        r = accept_type('ask("Q", on_parse_error = Abort())')
        assert len(r.warnings) == 1
        assert "on_parse_error" in r.warnings[0].message


# ---------------------------------------------------------------------------
# Seed environment
# ---------------------------------------------------------------------------


class TestSeedEnv:
    def test_seed_env_shares_types(self) -> None:
        r1 = accept_type("let x: int = 1\nx")
        prog2 = parse_program("x")
        r2 = check(
            resolve(prog2, parent_scope=r1.resolved.root_scope),
            default_capabilities(),
            seed_env=r1.type_env,
        )
        assert r2.resolved.program is not None

    def test_seed_env_shares_function_signatures(self) -> None:
        r1 = accept_type("def f(x: int) -> int = x\nf(1)")
        assert "f" in r1.function_signatures
        prog2 = parse_program("f(2)")
        r2 = check(
            resolve(prog2, parent_scope=r1.resolved.root_scope),
            default_capabilities(),
            seed_env=r1.type_env,
        )
        assert r2.resolved.program is not None


# ---------------------------------------------------------------------------
# CheckedProgram fields
# ---------------------------------------------------------------------------


class TestCheckedProgram:
    def test_node_types_populated(self) -> None:
        r = accept_type("let x = 42\nx")
        assert len(r.node_types) > 0

    def test_contract_specs_populated_for_ask(self) -> None:
        r = accept_type('ask("hello")')
        assert len(r.contract_specs) == 1

    def test_call_sites_populated(self) -> None:
        r = accept_type('ask("hello")')
        assert len(r.call_sites) == 1

    def test_warnings_empty_when_no_issues(self) -> None:
        r = accept_type("let x = 1\nx")
        assert len(r.warnings) == 0

    def test_function_signatures_populated(self) -> None:
        r = accept_type("def f(x: int) -> int = x\nf(1)")
        assert "f" in r.function_signatures

    def test_type_env_accessible(self) -> None:
        r = accept_type("let x = 1\nx")
        assert r.type_env is not None
        assert isinstance(r.type_env, TypeEnvironment)

    def test_resolved_accessible(self) -> None:
        r = accept_type("let x = 1\nx")
        assert r.resolved is not None


# ---------------------------------------------------------------------------
# Type repr / kind
# ---------------------------------------------------------------------------


class TestTypeReprAndKind:
    def test_text_repr(self) -> None:
        assert repr(TextType()) == "text"

    def test_int_repr(self) -> None:
        assert repr(IntType()) == "int"

    def test_decimal_repr(self) -> None:
        assert repr(DecimalType()) == "decimal"

    def test_bool_repr(self) -> None:
        assert repr(BoolType()) == "bool"

    def test_json_repr(self) -> None:
        assert repr(JsonType()) == "json"

    def test_unit_repr(self) -> None:
        assert repr(UnitType()) == "unit"

    def test_agent_repr(self) -> None:
        assert repr(AgentType()) == "agent"

    def test_bottom_repr(self) -> None:
        assert repr(BottomType()) == "bottom"

    def test_list_repr(self) -> None:
        assert repr(ListType(elem=IntType())) == "list[int]"

    def test_dict_repr(self) -> None:
        assert repr(DictType(value=TextType())) == "dict[text, text]"

    def test_record_repr(self) -> None:
        assert repr(RecordType(name="Point", fields={})) == "Point"

    def test_enum_repr(self) -> None:
        assert repr(EnumType(name="Color", variants={})) == "Color"

    def test_exception_repr(self) -> None:
        assert repr(ExceptionType(name="Abort", fields={})) == "Abort"

    def test_function_repr(self) -> None:
        ft = FunctionType(params=(IntType(), TextType()), result=BoolType())
        assert repr(ft) == "(int, text) -> bool"

    def test_function_no_params(self) -> None:
        ft = FunctionType(params=(), result=IntType())
        assert repr(ft) == "() -> int"

    def test_text_kind(self) -> None:
        assert TextType().kind == "text"

    def test_json_kind(self) -> None:
        assert JsonType().kind == "json"

    def test_bool_kind(self) -> None:
        assert BoolType().kind == "bool"

    def test_int_kind(self) -> None:
        assert IntType().kind == "int"

    def test_decimal_kind(self) -> None:
        assert DecimalType().kind == "decimal"

    def test_unit_kind(self) -> None:
        assert UnitType().kind == "unit"

    def test_agent_kind(self) -> None:
        assert AgentType().kind == "agent"

    def test_bottom_kind(self) -> None:
        assert BottomType().kind == "bottom"

    def test_list_kind(self) -> None:
        assert ListType(elem=IntType()).kind == "list"

    def test_dict_kind(self) -> None:
        assert DictType(value=IntType()).kind == "dict"

    def test_record_kind(self) -> None:
        assert RecordType(name="R", fields={}).kind == "record"

    def test_enum_kind(self) -> None:
        assert EnumType(name="E", variants={}).kind == "enum"

    def test_exception_kind(self) -> None:
        assert ExceptionType(name="Ex", fields={}).kind == "exception"

    def test_function_kind(self) -> None:
        assert FunctionType(params=(), result=IntType()).kind == "function"


# ---------------------------------------------------------------------------
# FunctionSignature
# ---------------------------------------------------------------------------


class TestFunctionSignature:
    def test_basic_signature(self) -> None:
        sig = FunctionSignature(
            params=(("x", IntType(), False), ("y", TextType(), True)),
            result=BoolType(),
        )
        assert sig.result == BoolType()
        assert sig.params[0] == ("x", IntType(), False)
        assert sig.params[1] == ("y", TextType(), True)

    def test_empty_params(self) -> None:
        sig = FunctionSignature(params=(), result=UnitType())
        assert sig.params == ()
        assert sig.result == UnitType()

    def test_frozen(self) -> None:
        sig = FunctionSignature(params=(), result=IntType())
        with pytest.raises((AttributeError, TypeError)):
            setattr(sig, "params", ())


# ---------------------------------------------------------------------------
# Config declaration (pass-through)
# ---------------------------------------------------------------------------


class TestConfigDeclPassthrough:
    def test_config_decl_accepted(self) -> None:
        r = accept_type("config log = true\n1")
        assert r.resolved.program is not None


# ---------------------------------------------------------------------------
# Miscellaneous / coverage-focused tests
# ---------------------------------------------------------------------------


class TestMisc:
    def test_all_node_ids_helper(self) -> None:
        prog = parse_program("let x = 1\nx")
        ids = all_node_ids(prog)
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_exception_base_is_abstract(self) -> None:
        assert EXCEPTION_BASE.abstract is True
        assert EXCEPTION_BASE.name == "Exception"

    def test_builtin_exceptions_in_names(self) -> None:
        assert "Abort" in BUILTIN_EXCEPTION_NAMES
        assert "AgentParseError" in BUILTIN_EXCEPTION_NAMES

    def test_prelude_types_exec_result(self) -> None:
        assert "ExecResult" in BUILTIN_PRELUDE_TYPES
        er = BUILTIN_PRELUDE_TYPES["ExecResult"]
        assert isinstance(er, RecordType)

    def test_prelude_types_parse_policy(self) -> None:
        assert "ParsePolicy" in BUILTIN_PRELUDE_TYPES
        pp = BUILTIN_PRELUDE_TYPES["ParsePolicy"]
        assert isinstance(pp, EnumType)
        assert "Abort" in pp.variants
        assert "Retry" in pp.variants

    def test_bottom_type_equality(self) -> None:
        assert BottomType() == BottomType()

    def test_is_test_simple(self) -> None:
        r = accept_type("enum E\n  | A\n  | B\nlet e = A()\ne is E.A")
        assert r.resolved.program is not None

    def test_template_empty_dict_in_template(self) -> None:
        # An empty dict inside a template needs annotation context
        r = accept_type('let d: json = {}\n"${d}"')
        assert r.resolved.program is not None

    def test_template_nested_json_in_dict(self) -> None:
        r = accept_type('let n = 42\n"${{"a": n}}"')
        assert r.resolved.program is not None

    def test_select_codec_no_match_raises(self) -> None:
        # text-only caps can't serve a record target
        err = reject_type(
            "record R\n  x: int\nlet r: R = ask(\"Q\")\nr",
            capabilities=text_only_caps(),
        )
        assert "codec" in str(err).lower() or "No registered" in str(err)

    def test_validate_format_option_unsupported_kind(self) -> None:
        # Codec 'text' doesn't support 'int' kind
        err = reject_type('let n: int = ask("Q", format = "text")\nn')
        assert "text" in str(err) or "support" in str(err).lower()

    def test_funcdef_value_binding_type(self) -> None:
        r = accept_type('def f(x: int) -> text = "hi"\nf')
        f_ref = r.resolved.program.body.items[1]
        assert isinstance(f_ref, VarRef)
        t = r.node_types[f_ref.node_id]
        assert isinstance(t, FunctionType)

    def test_param_no_annotation_is_text(self) -> None:
        r = accept_type("param x\nx")
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, ParamDecl)
        assert r.type_env.get_binding_type(decl.node_id) == TextType()

    def test_param_bottom_default_without_annotation_raises(self) -> None:
        err = reject_type('param x = raise Abort(message = "e")\nx')
        assert "Cannot infer type of param" in str(err)

    def test_do_loop_yields_unit(self) -> None:
        r = accept_type("var i = 0\ndo\n  i := i + 1\nuntil i > 5")
        do_node = r.resolved.program.body.items[1]
        assert isinstance(do_node, Do)
        assert r.node_types[do_node.node_id] == UnitType()

    def test_case_empty_branches_wildcard(self) -> None:
        r = accept_type('let x = 1\ncase x of | _ => "ok"')
        assert r.resolved.program is not None

    def test_qualified_enum_variant_wrong_qualifier_raises(self) -> None:
        err = reject_type(
            "enum A\n  | X\nenum B\n  | X\nlet a = A.X()\na is B.X"
        )
        assert "qualifier" in str(err).lower() or "enum" in str(err).lower()

    def test_catch_wildcard_binding_gets_base(self) -> None:
        # body and handler must unify; using text in both
        r = accept_type('try "hello" catch _ as e => e.message')
        assert r.resolved.program is not None

    def test_function_signature_ordering_valid(self) -> None:
        r = accept_type('def f(x: int, y: text = "ok") -> int = x\nf(1)')
        sig = r.function_signatures["f"]
        assert sig.params[0].has_default is False   # x: not defaulted
        assert sig.params[1].has_default is True    # y: has default

    def test_ask_strict_json_false(self) -> None:
        r = accept_type('let n: int = ask("Q", format = "json", strict_json = false)\nn')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        spec = r.contract_specs[decl.value.node_id]
        assert spec.strict_json is False

    def test_in_op_dict_right_operand(self) -> None:
        r = accept_type('let d: dict[text, int] = {"k": 1}\n"k" in d')
        assert r.resolved.program is not None

    def test_in_op_invalid_raises(self) -> None:
        err = reject_type("1 in 2")
        assert "in" in str(err).lower()

    def test_constructor_pattern_duplicate_field_raises(self) -> None:
        err = reject_type(
            "enum E\n  | A(x: int)\nlet e = A(x = 1)\n"
            "case e of | E.A(x = n, x = m) => n | _ => 0"
        )
        assert "duplicate" in str(err).lower() or "field" in str(err).lower()

    def test_constructor_pattern_unknown_field_raises(self) -> None:
        err = reject_type(
            "enum E\n  | A(x: int)\nlet e = A(x = 1)\n"
            "case e of | E.A(z = n) => n | _ => 0"
        )
        msg = str(err).lower()
        assert "no field" in msg or "unknown" in msg or "field" in msg

    def test_variant_qualifier_wrong_raises(self) -> None:
        err = reject_type(
            "enum A\n  | X\nenum B\n  | X\nlet a = A.X()\na is B.X"
        )
        assert "qualifier" in str(err).lower() or "enum" in str(err).lower()

    def test_qualified_constructor_wrong_enum_raises(self) -> None:
        err = reject_type("enum A\n  | X\nenum B\n  | Y\nA.Y()")
        assert "variant" in str(err).lower()

    def test_qualified_constructor_not_enum_raises(self) -> None:
        err = reject_type("record R\n  x: int\nR.Something()")
        assert "enum" in str(err).lower()

    def test_enum_variant_with_fields(self) -> None:
        r = accept_type(
            "enum Result\n  | Ok(value: int)\n  | Err(msg: text)\nOk(value = 42)"
        )
        assert r.resolved.program is not None

    def test_enum_variant_with_fields_case(self) -> None:
        r = accept_type(
            "enum Result\n  | Ok(value: int)\n  | Err(msg: text)\n"
            "let res = Ok(value = 42)\n"
            "case res of | Result.Ok(value = v) => v | Result.Err(msg = m) => 0"
        )
        assert r.resolved.program is not None

    def test_exec_result_contract_spec(self) -> None:
        r = accept_type('exec("ls")')
        call = r.resolved.program.body.items[0]
        assert isinstance(call, Call)
        spec = r.contract_specs[call.node_id]
        assert spec.structured_exec is True

    def test_agent_decl_is_agent_type(self) -> None:
        r = accept_type("agent a\na")
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, AgentDecl)
        assert r.type_env.get_binding_type(decl.node_id) == AgentType()

    def test_builtin_prelude_type_names_coverage(self) -> None:
        assert len(BUILTIN_PRELUDE_TYPE_NAMES) > 0
        for name in BUILTIN_PRELUDE_TYPE_NAMES:
            assert isinstance(name, str)

    def test_record_field_types(self) -> None:
        r = accept_type("record P\n  x: int\n  y: decimal\nP(x = 1, y = 2.5)")
        assert r.resolved.program is not None

    def test_builtin_type_name_shadow_raises(self) -> None:
        # ExecResult is a BUILTIN_PRELUDE_TYPE_NAMES — record shadows it
        err = reject_type("record ExecResult\n  x: int\nExecResult(x = 1)")
        assert "built-in" in str(err).lower() or "ExecResult" in str(err)

    def test_alias_to_record_field(self) -> None:
        # Exercises alias->record resolution in _ensure_referenced_type_built
        r = accept_type("record R\n  x: int\ntype MyR = R\nlet r: MyR = R(x = 1)\nr")
        assert r.resolved.program is not None

    def test_field_list_of_record(self) -> None:
        # Exercises ListT path in _ensure_referenced_type_built
        r = accept_type(
            "record R\n  x: int\nrecord S\n  items: list[R]\nS(items = [])"
        )
        assert r.resolved.program is not None

    def test_template_empty_list_in_dict_value(self) -> None:
        # Exercises empty-list in _check_template_literal_child
        r = accept_type('"${{ "a": []}}"')
        assert r.resolved.program is not None

    def test_template_empty_dict_as_list_child(self) -> None:
        # Exercises empty-dict in _check_template_literal_child
        r = accept_type('"${[{}]}"')
        assert r.resolved.program is not None

    def test_template_dup_key_in_interp_dict(self) -> None:
        err = reject_type('"${{"a": 1, "a": 2}}"')
        assert "duplicate" in str(err).lower() or "key" in str(err).lower()

    def test_is_test_with_correct_qualifier(self) -> None:
        # Exercises the qualifier check path in is-test expressions.
        r = accept_type("enum E\n  | A\n  | B\nlet e = E.A()\ne is E.A")
        assert r.resolved.program is not None

    def test_is_test_qualifier_not_enum_raises(self) -> None:
        # Exercises the error path when the qualifier resolves to a non-enum type.
        err = reject_type("enum A\n  | X\nrecord R\n  x: int\nlet a = A.X()\na is R.X")
        assert "not a known enum" in str(err).lower() or "enum" in str(err).lower()

    def test_is_test_unknown_qualifier_raises(self) -> None:
        # Exercises the error path when the qualifier name is not a known enum.
        err = reject_type("enum E\n  | A\nlet e = E.A()\ne is UnknownEnum.A")
        assert "not a known enum" in str(err).lower() or "enum" in str(err).lower()

    def test_enum_variant_field_duplicate_raises(self) -> None:
        # Exercises line 287: duplicate field in enum variant
        err = reject_type("enum E\n  | A(x: int, x: text)\nA(x = 1)")
        assert "duplicate" in str(err).lower() or "field" in str(err).lower()

    def test_unqualified_ctor_with_enum_expected_type(self) -> None:
        # Exercises unqualified constructor disambiguation when an expected EnumType is present.
        r = accept_type("enum E\n  | A\n  | B\nlet x: E = A()\nx")
        assert r.resolved.program is not None

    def test_branch_decimal_int_widening(self) -> None:
        # if true => 2.5 | else => 2 → decimal (decimal+int branch unification widens to decimal)
        r = accept_type("if true => 2.5 | true => 3 | else => 1.0")
        if_node = r.resolved.program.body.items[0]
        assert isinstance(if_node, If)
        t = r.node_types[if_node.node_id]
        assert t == DecimalType()

    def test_constructor_pattern_with_qualifier(self) -> None:
        # Exercises the qualifier check in constructor pattern matching.
        r = accept_type(
            "enum E\n  | A(x: int)\nlet e = A(x = 1)\n"
            "case e of | E.A(x = n) => n | _ => 0"
        )
        assert r.resolved.program is not None

    def test_constructor_pattern_wrong_variant_raises(self) -> None:
        # Exercises the error when a constructor pattern variant is not found in the enum.
        err = reject_type(
            "enum E\n  | A\n  | B\nlet e = A()\n"
            "case e of | E.C() => 1 | _ => 0"
        )
        assert "variant" in str(err).lower()

    def test_env_resolve_named_type_via_alias(self) -> None:
        # Exercises get_alias_target_expr and resolve_named_type alias chain
        env = TypeEnvironment()
        from agm.agl.syntax.types import NameT
        sp = mk_span()
        env.register_type("R", RecordType(name="R", fields={}))
        env.register_alias("MyR", NameT(name="R", span=sp, node_id=1))
        result = env.resolve_named_type("MyR")
        assert result == RecordType(name="R", fields={})

    def test_env_resolve_named_type_with_bad_alias(self) -> None:
        # Exercises the except AglTypeError: return None path in resolve_named_type
        env = TypeEnvironment()
        from agm.agl.syntax.types import NameT
        sp = mk_span()
        # Register a cycle to cause AglTypeError internally
        env.register_alias("A", NameT(name="B", span=sp, node_id=1))
        env.register_alias("B", NameT(name="A", span=sp, node_id=2))
        result = env.resolve_named_type("A")
        assert result is None

    def test_retry_with_non_int_n_raises(self) -> None:
        # Exercises line 880->879: Retry n_arg not an IntLit
        err = reject_type('let n: int = ask("Q", on_parse_error = Retry(n = "bad"))\nn')
        assert "on_parse_error" in str(err).lower() or "Retry" in str(err)

    def test_retry_with_wrong_key_raises(self) -> None:
        # Exercises line 880 -> falls through to raise
        err = reject_type('let n: int = ask("Q", on_parse_error = Retry(m = 3))\nn')
        assert "on_parse_error" in str(err).lower() or "Retry" in str(err)

    def test_parse_policy_unknown_variant_raises(self) -> None:
        # Exercises line 877->890: arg.name is neither "Abort" nor "Retry"
        err = reject_type('let n: int = ask("Q", on_parse_error = ParsePolicy.Bad())\nn')
        assert "on_parse_error" in str(err).lower() or "ParsePolicy" in str(err)

    def test_exec_strict_json_non_bool_raises(self) -> None:
        # Exercises line 815: strict_json non-BoolLit in exec
        err = reject_type('let n: int = exec("ls", format = "json", strict_json = "yes")\nn')
        assert "strict_json" in str(err).lower() or "bool" in str(err).lower()

    def test_decimal_subtraction_yields_decimal(self) -> None:
        # Exercises _check_numeric_binop returning DecimalType for decimal subtraction.
        r = accept_type("1.5 - 0.5")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == DecimalType()

    def test_decimal_multiplication_yields_decimal(self) -> None:
        # Also exercises the decimal path in _check_numeric_binop.
        r = accept_type("2.0 * 3.0")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == DecimalType()

    def test_list_decimal_then_int_widening(self) -> None:
        # Exercises list element type unification: decimal followed by int widens to decimal.
        # is_assignable(decimal,int)=False but is_assignable(int,decimal)=True, so type widens.
        r = accept_type("[2.5, 1]")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == ListType(elem=DecimalType())

    def test_alias_field_in_record(self) -> None:
        # Exercises lines 314-316: alias in _ensure_referenced_type_built
        r = accept_type("type N = int\nrecord R\n  x: N\nR(x = 1)")
        assert r.resolved.program is not None

    def test_list_field_of_aliased_type(self) -> None:
        # Exercises line 321: ListT in _ensure_referenced_type_built
        r = accept_type("type N = int\nrecord R\n  xs: list[N]\nR(xs = [])")
        assert r.resolved.program is not None

    def test_dict_field_of_aliased_type(self) -> None:
        # Exercises line 323: DictT in _ensure_referenced_type_built
        r = accept_type("type N = int\nrecord R\n  d: dict[text, N]\nR(d = {})")
        assert r.resolved.program is not None

    def test_two_records_same_enum_field(self) -> None:
        # Exercises line 246: _ensure_built_enum called twice returns early
        r = accept_type("enum E\n  | A\nrecord R1\n  e: E\nrecord R2\n  e: E\nR1(e = A())")
        assert r.resolved.program is not None

    def test_template_nested_list_in_list(self) -> None:
        # Exercises a non-empty ListLit as a child of a template list.
        r = accept_type('"${[1, [2, 3]]}"')
        assert r.resolved.program is not None

    def test_template_nested_dict_in_dict(self) -> None:
        # Exercises a non-empty DictLit as a child of a template dict.
        r = accept_type('"${{"a": {"b": 1}}}"')
        assert r.resolved.program is not None

    def test_is_test_without_qualifier(self) -> None:
        # Exercises the is-test without a qualifier (no qualifier check is performed).
        r = accept_type("enum E\n  | A\n  | B\nlet e = E.A()\ne is A")
        assert r.resolved.program is not None

    def test_constructor_pattern_without_qualifier(self) -> None:
        # Exercises a constructor pattern without a qualifier.
        r = accept_type(
            "enum E\n  | A(x: int)\nlet e = A(x = 1)\n"
            "case e of | A(x = n) => n | _ => 0"
        )
        assert r.resolved.program is not None

    def test_record_field_of_builtin_record_type(self) -> None:
        # Exercises 314->exit: NameT("ExecResult") is in env but not in _record_defs
        # (it's a built-in type, not user-declared)
        r = accept_type(
            "record Wrapper\n  result: ExecResult\n"
            "let x = exec(\"ls\")\nWrapper(result = x)"
        )
        assert r.resolved.program is not None

    def test_funcdef_builtin_type_name_rejected(self) -> None:
        # Exercises the error when a def is named after a built-in type.
        # (scope does not reject 'text'/'int'/etc. as def names, only print/exec/ask)
        err = reject_type("def text() -> int = 1\ntext()")
        assert "built-in type name" in str(err)

    def test_all_bottom_if_branches_yield_bottom(self) -> None:
        # Exercises _unify_branch_types returning BottomType when all branches always raise.
        r = accept_type(
            "def f(x: int) -> int =\n"
            "  if x == 0 =>\n"
            "    let msg = \"zero\"\n"
            "    raise Abort(message = msg)\n"
            "  | else =>\n"
            "    let msg = \"nonzero\"\n"
            "    raise Abort(message = msg)"
        )
        assert r.resolved.program is not None


class TestIndexTypechecking:
    def _binding_ref(
        self,
        name: str,
        *,
        mutable: bool,
        decl_node_id: int,
        kind: BinderKind,
    ) -> BindingRef:
        return BindingRef(
            name=name,
            mutable=mutable,
            decl_span=mk_span(),
            decl_node_id=decl_node_id,
            kind=kind,
        )

    def _check_items(
        self,
        items: tuple[Item, ...],
        resolution: dict[int, BindingRef],
    ) -> CheckedProgram:
        sp = mk_span()
        block = Block(items=items, span=sp, node_id=_mk_node_id())
        program = Program(body=block, span=sp, node_id=_mk_node_id())
        resolved = _ResolvedProgram(
            program=program,
            resolution=resolution,
            builtin_calls={},
            root_scope=ScopeNode(node_id=program.node_id),
        )
        return check(resolved, default_capabilities())

    def _list_decl_and_ref(
        self, *, mutable: bool = False
    ) -> tuple[LetDecl | VarDecl, VarRef, BindingRef]:
        sp = mk_span()
        decl_cls = VarDecl if mutable else LetDecl
        decl = decl_cls(
            name="xs",
            type_ann=ListT(
                elem=IntT(span=sp, node_id=_mk_node_id()),
                span=sp,
                node_id=_mk_node_id(),
            ),
            value=ListLit(
                elements=(
                    IntLit(value=10, span=sp, node_id=_mk_node_id()),
                    IntLit(value=20, span=sp, node_id=_mk_node_id()),
                ),
                span=sp,
                node_id=_mk_node_id(),
            ),
            span=sp,
            node_id=_mk_node_id(),
        )
        ref_expr = VarRef(name="xs", span=sp, node_id=_mk_node_id())
        ref = self._binding_ref(
            "xs",
            mutable=mutable,
            decl_node_id=decl.node_id,
            kind=BinderKind.var_binding if mutable else BinderKind.let_binding,
        )
        return decl, ref_expr, ref

    def _dict_decl_and_ref(
        self, *, mutable: bool = False
    ) -> tuple[LetDecl | VarDecl, VarRef, BindingRef]:
        sp = mk_span()
        decl_cls = VarDecl if mutable else LetDecl
        decl = decl_cls(
            name="d",
            type_ann=DictT(
                value=IntT(span=sp, node_id=_mk_node_id()),
                span=sp,
                node_id=_mk_node_id(),
            ),
            value=DictLit(
                entries=(
                    DictEntry(
                        key=StringLit(value="a", span=sp, node_id=_mk_node_id()),
                        value=IntLit(value=1, span=sp, node_id=_mk_node_id()),
                        span=sp,
                        node_id=_mk_node_id(),
                    ),
                ),
                span=sp,
                node_id=_mk_node_id(),
            ),
            span=sp,
            node_id=_mk_node_id(),
        )
        ref_expr = VarRef(name="d", span=sp, node_id=_mk_node_id())
        ref = self._binding_ref(
            "d",
            mutable=mutable,
            decl_node_id=decl.node_id,
            kind=BinderKind.var_binding if mutable else BinderKind.let_binding,
        )
        return decl, ref_expr, ref

    def test_list_index_returns_element_type(self) -> None:
        sp = mk_span()
        decl, obj, ref = self._list_decl_and_ref()
        index = IndexAccess(
            obj=obj,
            index=IntLit(value=0, span=sp, node_id=_mk_node_id()),
            span=sp,
            node_id=_mk_node_id(),
        )
        result = self._check_items((decl, cast(Item, index)), {obj.node_id: ref})
        assert result.node_types[index.node_id] == IntType()

    def test_dict_index_returns_value_type(self) -> None:
        sp = mk_span()
        decl, obj, ref = self._dict_decl_and_ref()
        index = IndexAccess(
            obj=obj,
            index=StringLit(value="a", span=sp, node_id=_mk_node_id()),
            span=sp,
            node_id=_mk_node_id(),
        )
        result = self._check_items((decl, cast(Item, index)), {obj.node_id: ref})
        assert result.node_types[index.node_id] == IntType()

    def test_bad_index_operands_and_non_container_rejected(self) -> None:
        sp = mk_span()
        list_decl, list_obj, list_ref = self._list_decl_and_ref()
        list_index = IndexAccess(
            obj=list_obj,
            index=StringLit(value="a", span=sp, node_id=_mk_node_id()),
            span=sp,
            node_id=_mk_node_id(),
        )
        with pytest.raises(AglTypeError, match="expected 'int'"):
            self._check_items((list_decl, cast(Item, list_index)), {list_obj.node_id: list_ref})

        dict_decl, dict_obj, dict_ref = self._dict_decl_and_ref()
        dict_index = IndexAccess(
            obj=dict_obj,
            index=IntLit(value=0, span=sp, node_id=_mk_node_id()),
            span=sp,
            node_id=_mk_node_id(),
        )
        with pytest.raises(AglTypeError, match="expected 'text'"):
            self._check_items((dict_decl, cast(Item, dict_index)), {dict_obj.node_id: dict_ref})

        decl = LetDecl(
            name="n",
            type_ann=IntT(span=sp, node_id=_mk_node_id()),
            value=IntLit(value=1, span=sp, node_id=_mk_node_id()),
            span=sp,
            node_id=_mk_node_id(),
        )
        obj = VarRef(name="n", span=sp, node_id=_mk_node_id())
        ref = self._binding_ref(
            "n",
            mutable=False,
            decl_node_id=decl.node_id,
            kind=BinderKind.let_binding,
        )
        non_container = IndexAccess(
            obj=obj,
            index=IntLit(value=0, span=sp, node_id=_mk_node_id()),
            span=sp,
            node_id=_mk_node_id(),
        )
        with pytest.raises(AglTypeError, match="list or dict"):
            self._check_items((decl, cast(Item, non_container)), {obj.node_id: ref})

    def test_parsed_indexed_assignment_accepts_var_list(self) -> None:
        result = accept_type("var xs = [1]\nxs[0] := 2\nxs")
        program = result.resolved.program
        assert program is not None
        final_expr = program.body.items[2]
        assert isinstance(final_expr, VarRef)
        assert result.node_types[final_expr.node_id] == ListType(elem=IntType())

    def test_parsed_chained_indexed_assignment_records_intermediate_target_type(self) -> None:
        result = accept_type("var matrix = [[1, 2]]\nmatrix[0][1] := 3\nmatrix")
        program = result.resolved.program
        assert program is not None

        assign_stmt = program.body.items[1]
        assert isinstance(assign_stmt, AssignStmt)
        target = assign_stmt.target
        assert isinstance(target, IndexTarget)
        intermediate = target.obj
        assert isinstance(intermediate, IndexAccess)
        assert result.node_types[intermediate.node_id] == ListType(elem=IntType())

        final_expr = program.body.items[2]
        assert isinstance(final_expr, VarRef)
        assert result.node_types[final_expr.node_id] == ListType(elem=ListType(elem=IntType()))

    def test_parsed_indexed_assignment_rejects_non_container_root(self) -> None:
        err = reject_type("var n = 1\nn[0] := 2\nn")
        assert "list or dict" in str(err)

    def test_indexed_assignment_accepts_var_list_and_dict(self) -> None:
        sp = mk_span()
        list_decl, list_obj, list_ref = self._list_decl_and_ref(mutable=True)
        list_assign = AssignStmt(
            target=IndexTarget(
                obj=list_obj,
                index=IntLit(value=0, span=sp, node_id=_mk_node_id()),
                span=sp,
                node_id=_mk_node_id(),
            ),
            value=IntLit(value=3, span=sp, node_id=_mk_node_id()),
            span=sp,
            node_id=_mk_node_id(),
        )
        self._check_items(
            (list_decl, list_assign),
            {list_obj.node_id: list_ref, list_assign.node_id: list_ref},
        )

        dict_decl, dict_obj, dict_ref = self._dict_decl_and_ref(mutable=True)
        dict_assign = AssignStmt(
            target=IndexTarget(
                obj=dict_obj,
                index=StringLit(value="a", span=sp, node_id=_mk_node_id()),
                span=sp,
                node_id=_mk_node_id(),
            ),
            value=IntLit(value=3, span=sp, node_id=_mk_node_id()),
            span=sp,
            node_id=_mk_node_id(),
        )
        self._check_items(
            (dict_decl, dict_assign),
            {dict_obj.node_id: dict_ref, dict_assign.node_id: dict_ref},
        )

    def test_indexed_assignment_rejects_immutable_and_invalid_roots(self) -> None:
        sp = mk_span()
        decl, obj, ref = self._list_decl_and_ref()
        let_assign = AssignStmt(
            target=IndexTarget(
                obj=obj,
                index=IntLit(value=0, span=sp, node_id=_mk_node_id()),
                span=sp,
                node_id=_mk_node_id(),
            ),
            value=IntLit(value=3, span=sp, node_id=_mk_node_id()),
            span=sp,
            node_id=_mk_node_id(),
        )
        with pytest.raises(AglTypeError, match="mutable 'var'"):
            self._check_items((decl, let_assign), {obj.node_id: ref, let_assign.node_id: ref})

        mutable_decl, mutable_obj, mutable_ref = self._list_decl_and_ref(mutable=True)
        field_assign = AssignStmt(
            target=IndexTarget(
                obj=FieldAccess(
                    obj=mutable_obj,
                    field="field",
                    span=sp,
                    node_id=_mk_node_id(),
                ),
                index=IntLit(value=0, span=sp, node_id=_mk_node_id()),
                span=sp,
                node_id=_mk_node_id(),
            ),
            value=IntLit(value=3, span=sp, node_id=_mk_node_id()),
            span=sp,
            node_id=_mk_node_id(),
        )
        with pytest.raises(AglTypeError, match="variable list or dict root"):
            self._check_items(
                (mutable_decl, field_assign),
                {
                    mutable_obj.node_id: mutable_ref,
                    field_assign.node_id: mutable_ref,
                },
            )

        temporary_assign = AssignStmt(
            target=IndexTarget(
                obj=ListLit(
                    elements=(IntLit(value=1, span=sp, node_id=_mk_node_id()),),
                    span=sp,
                    node_id=_mk_node_id(),
                ),
                index=IntLit(value=0, span=sp, node_id=_mk_node_id()),
                span=sp,
                node_id=_mk_node_id(),
            ),
            value=IntLit(value=3, span=sp, node_id=_mk_node_id()),
            span=sp,
            node_id=_mk_node_id(),
        )
        with pytest.raises(AglTypeError, match="variable list or dict root"):
            self._check_items(
                (mutable_decl, temporary_assign),
                {temporary_assign.node_id: mutable_ref},
            )

    def test_indexed_assignment_rejects_param_function_arg_and_non_container(self) -> None:
        sp = mk_span()
        param = ParamDecl(
            name="xs",
            annotation=ListT(
                elem=IntT(span=sp, node_id=_mk_node_id()),
                span=sp,
                node_id=_mk_node_id(),
            ),
            default=None,
            span=sp,
            node_id=_mk_node_id(),
        )
        param_obj = VarRef(name="xs", span=sp, node_id=_mk_node_id())
        param_ref = self._binding_ref(
            "xs",
            mutable=False,
            decl_node_id=param.node_id,
            kind=BinderKind.param_binding,
        )
        param_assign = AssignStmt(
            target=IndexTarget(
                obj=param_obj,
                index=IntLit(value=0, span=sp, node_id=_mk_node_id()),
                span=sp,
                node_id=_mk_node_id(),
            ),
            value=IntLit(value=3, span=sp, node_id=_mk_node_id()),
            span=sp,
            node_id=_mk_node_id(),
        )
        with pytest.raises(AglTypeError, match="mutable 'var'"):
            self._check_items(
                (param, param_assign),
                {param_obj.node_id: param_ref, param_assign.node_id: param_ref},
            )

        fn_param = Param(
            name="arg",
            type_expr=ListT(
                elem=IntT(span=sp, node_id=_mk_node_id()),
                span=sp,
                node_id=_mk_node_id(),
            ),
            kind=ParamKind.STANDARD,
            default=None,
            span=sp,
            node_id=_mk_node_id(),
        )
        arg_obj = VarRef(name="arg", span=sp, node_id=_mk_node_id())
        arg_ref = self._binding_ref(
            "arg",
            mutable=False,
            decl_node_id=fn_param.node_id,
            kind=BinderKind.param_binding,
        )
        arg_assign = AssignStmt(
            target=IndexTarget(
                obj=arg_obj,
                index=IntLit(value=0, span=sp, node_id=_mk_node_id()),
                span=sp,
                node_id=_mk_node_id(),
            ),
            value=IntLit(value=3, span=sp, node_id=_mk_node_id()),
            span=sp,
            node_id=_mk_node_id(),
        )
        fn = FuncDef(
            name="f",
            params=(fn_param,),
            return_type=UnitT(span=sp, node_id=_mk_node_id()),
            body=Block(
                items=(arg_assign, UnitLit(span=sp, node_id=_mk_node_id())),
                span=sp,
                node_id=_mk_node_id(),
            ),
            span=sp,
            node_id=_mk_node_id(),
        )
        with pytest.raises(AglTypeError, match="mutable 'var'"):
            self._check_items((fn,), {arg_obj.node_id: arg_ref, arg_assign.node_id: arg_ref})

        var_decl = VarDecl(
            name="n",
            type_ann=IntT(span=sp, node_id=_mk_node_id()),
            value=IntLit(value=1, span=sp, node_id=_mk_node_id()),
            span=sp,
            node_id=_mk_node_id(),
        )
        var_obj = VarRef(name="n", span=sp, node_id=_mk_node_id())
        var_ref = self._binding_ref(
            "n",
            mutable=True,
            decl_node_id=var_decl.node_id,
            kind=BinderKind.var_binding,
        )
        non_container_assign = AssignStmt(
            target=IndexTarget(
                obj=var_obj,
                index=IntLit(value=0, span=sp, node_id=_mk_node_id()),
                span=sp,
                node_id=_mk_node_id(),
            ),
            value=IntLit(value=3, span=sp, node_id=_mk_node_id()),
            span=sp,
            node_id=_mk_node_id(),
        )
        with pytest.raises(AglTypeError, match="list or dict"):
            self._check_items(
                (var_decl, non_container_assign),
                {var_obj.node_id: var_ref, non_container_assign.node_id: var_ref},
            )

    def test_indexed_assignment_value_type_mismatch_rejected(self) -> None:
        sp = mk_span()
        decl, obj, ref = self._list_decl_and_ref(mutable=True)
        stmt = AssignStmt(
            target=IndexTarget(
                obj=obj,
                index=IntLit(value=0, span=sp, node_id=_mk_node_id()),
                span=sp,
                node_id=_mk_node_id(),
            ),
            value=StringLit(value="nope", span=sp, node_id=_mk_node_id()),
            span=sp,
            node_id=_mk_node_id(),
        )
        with pytest.raises(AglTypeError, match="expected 'int'"):
            self._check_items((decl, stmt), {obj.node_id: ref, stmt.node_id: ref})

    def test_invalid_direct_ast_assign_target_rejected(self) -> None:
        sp = mk_span()
        stmt = AssignStmt(
            target=cast(AssignTarget, UnitLit(span=sp, node_id=_mk_node_id())),
            value=IntLit(value=1, span=sp, node_id=_mk_node_id()),
            span=sp,
            node_id=_mk_node_id(),
        )
        with pytest.raises(AglTypeError, match="assignment target"):
            self._check_items((stmt,), {})


# ---------------------------------------------------------------------------
# TestDefensiveGuards — direct AST construction to cover defensive branches
# ---------------------------------------------------------------------------


class TestDefensiveGuards:
    """Cover defensive guards that are unreachable from the parser.

    These tests construct AST nodes and ``ResolvedProgram`` objects directly to
    exercise branches that the parser/scope pass prevent from being reached via
    normal source code.  This ensures 100% branch coverage of the checker.
    """

    def _mk_resolved(
        self,
        program: Program,
        resolution: dict[int, BindingRef] | None = None,
        builtin_calls: dict[int, object] | None = None,
        declared_functions: dict[str, FuncDef] | None = None,
    ) -> _ResolvedProgram:
        from agm.agl.scope.symbols import BuiltinKind as _BuiltinKind

        root = ScopeNode(node_id=program.node_id)
        bc: dict[int, _BuiltinKind] = {}
        return _ResolvedProgram(
            program=program,
            resolution=resolution or {},
            builtin_calls=bc,
            root_scope=root,
            declared_functions=declared_functions or {},
        )

    def test_empty_block_yields_unit(self) -> None:
        # Exercises the path where _check_block returns UnitType() for an empty block.
        # The grammar never produces an empty block from source, so we construct
        # the AST directly.
        sp = mk_span()
        block = Block(items=(), span=sp, node_id=_mk_node_id())
        prog = Program(body=block, span=sp, node_id=_mk_node_id())
        resolved = self._mk_resolved(prog)
        result = check(resolved, default_capabilities())
        assert result is not None

    def test_empty_case_branches_fallback(self) -> None:
        # Exercises the path where _check_case returns fallback type when branches is empty.
        # The grammar requires at least one branch, so we construct directly.
        sp = mk_span()
        subject = IntLit(value=1, span=sp, node_id=_mk_node_id())
        case_node = Case(subject=subject, branches=(), span=sp, node_id=_mk_node_id())
        block = Block(items=(case_node,), span=sp, node_id=_mk_node_id())
        prog = Program(body=block, span=sp, node_id=_mk_node_id())
        resolved = self._mk_resolved(prog)
        result = check(resolved, default_capabilities())
        assert result is not None

    def test_duplicate_constructor_arg_rejected(self) -> None:
        # Parser rejects duplicate named args at parse time (AglSyntaxError/AglTypeError).
        # Use reject_any since the parser may catch it before the type checker.
        err = reject_any("record P\n  x: int\nP(x = 1, x = 2)")
        assert "duplicate" in str(err).lower() or "x" in str(err)

    def test_builtin_func_name_def_rejected(self) -> None:
        # Exercises line 372: _preregister_funcdef raises for names in _BUILTIN_FUNC_NAMES.
        # The scope pass rejects print/exec/ask before typecheck, so we bypass it
        # by building a FuncDef node with name "print" inside a ResolvedProgram.
        sp = mk_span()
        body_expr = IntLit(value=1, span=sp, node_id=_mk_node_id())
        ret_type = IntT(span=sp, node_id=_mk_node_id())
        fd = FuncDef(
            name="print", params=(), return_type=ret_type, body=body_expr,
            span=sp, node_id=_mk_node_id(),
        )
        block = Block(items=(fd,), span=sp, node_id=_mk_node_id())
        prog = Program(body=block, span=sp, node_id=_mk_node_id())
        resolved = self._mk_resolved(prog, declared_functions={"print": fd})
        with pytest.raises(AglTypeError, match="built-in function"):
            check(resolved, default_capabilities())

    def test_alias_seen_guard_in_ensure_referenced(self) -> None:
        err = reject_type(
            "record Wrapper\n"
            "  value: A\n"
            "type A = B\n"
            "type B = A\n"
            "()"
        )
        assert "cycle" in str(err).lower()
        assert "a" in str(err).lower()

    def test_binding_type_not_set_assertion(self) -> None:
        # Exercises line 609: _require_binding_type raises AssertionError when a
        # VarRef resolves to a BindingRef whose decl_node_id has no type in the env.
        sp = mk_span()
        decl_nid = _mk_node_id()
        ref_nid = _mk_node_id()
        varref = VarRef(name="x", span=sp, node_id=ref_nid)
        binding_ref = BindingRef(
            name="x", mutable=False, decl_span=sp, decl_node_id=decl_nid,
            kind=BinderKind.let_binding,
        )
        block = Block(items=(varref,), span=sp, node_id=_mk_node_id())
        prog = Program(body=block, span=sp, node_id=_mk_node_id())
        resolved = self._mk_resolved(prog, resolution={ref_nid: binding_ref})
        with pytest.raises(AssertionError, match="checker invariant"):
            check(resolved, default_capabilities())

    def test_declared_call_sig_none_fallback(self) -> None:
        # A function binding without a registered signature now reports a user-facing
        # inference error instead of falling through to an internal assertion.
        sp = mk_span()
        body_expr = IntLit(value=1, span=sp, node_id=_mk_node_id())
        ret_type = IntT(span=sp, node_id=_mk_node_id())
        fd_nid = _mk_node_id()
        fd = FuncDef(
            name="h", params=(), return_type=ret_type, body=body_expr,
            span=sp, node_id=fd_nid,
        )
        callee_nid = _mk_node_id()
        callee = VarRef(name="h", span=sp, node_id=callee_nid)
        call = Call(callee=callee, args=(), named_args=(), span=sp, node_id=_mk_node_id())
        # block has only the call — no FuncDef, so pre-pass skips h
        block = Block(items=(call,), span=sp, node_id=_mk_node_id())
        prog = Program(body=block, span=sp, node_id=_mk_node_id())
        binding_ref = BindingRef(
            name="h", mutable=False, decl_span=sp, decl_node_id=fd_nid,
            kind=BinderKind.function_binding,
        )
        resolved = self._mk_resolved(
            prog,
            resolution={callee_nid: binding_ref},
            declared_functions={"h": fd},
        )
        with pytest.raises(AglTypeError, match="Cannot infer return type"):
            check(resolved, default_capabilities())

    def test_function_value_without_registered_type_reports_inference_error(self) -> None:
        sp = mk_span()
        body_expr = IntLit(value=1, span=sp, node_id=_mk_node_id())
        ret_type = IntT(span=sp, node_id=_mk_node_id())
        fd_nid = _mk_node_id()
        fd = FuncDef(
            name="h",
            params=(),
            return_type=ret_type,
            body=body_expr,
            span=sp,
            node_id=fd_nid,
        )
        ref_nid = _mk_node_id()
        ref = VarRef(name="h", span=sp, node_id=ref_nid)
        block = Block(items=(ref,), span=sp, node_id=_mk_node_id())
        prog = Program(body=block, span=sp, node_id=_mk_node_id())
        binding_ref = BindingRef(
            name="h",
            mutable=False,
            decl_span=sp,
            decl_node_id=fd_nid,
            kind=BinderKind.function_binding,
        )
        resolved = self._mk_resolved(
            prog,
            resolution={ref_nid: binding_ref},
            declared_functions={"h": fd},
        )

        with pytest.raises(AglTypeError, match="Cannot infer return type"):
            check(resolved, default_capabilities())

    def test_builtin_funcdef_without_return_type_rejected_defensively(self) -> None:
        sp = mk_span()
        fd = FuncDef(
            name="print",
            params=(),
            return_type=None,
            body=None,
            span=sp,
            node_id=_mk_node_id(),
            is_builtin=True,
        )
        block = Block(items=(fd,), span=sp, node_id=_mk_node_id())
        prog = Program(body=block, span=sp, node_id=_mk_node_id())
        resolved = self._mk_resolved(prog, declared_functions={"print": fd})

        with pytest.raises(AglTypeError, match="must declare a return type"):
            check(resolved, default_capabilities())

    def test_duplicate_named_arg_in_declared_call(self) -> None:
        # Exercises line 970: duplicate named arg check in _check_declared_name_call.
        # The parser rejects duplicate named args, so we construct directly.
        sp = mk_span()
        p_nid = _mk_node_id()
        ret_t = IntT(span=sp, node_id=_mk_node_id())
        param_t = IntT(span=sp, node_id=_mk_node_id())
        param = Param(
            name="x", type_expr=param_t, kind=ParamKind.STANDARD, default=None,
            span=sp, node_id=p_nid
        )
        body_expr = IntLit(value=1, span=sp, node_id=_mk_node_id())
        fd_nid = _mk_node_id()
        fd = FuncDef(
            name="g", params=(param,), return_type=ret_t, body=body_expr,
            span=sp, node_id=fd_nid,
        )
        callee_nid = _mk_node_id()
        callee = VarRef(name="g", span=sp, node_id=callee_nid)
        val1 = IntLit(value=1, span=sp, node_id=_mk_node_id())
        val2 = IntLit(value=2, span=sp, node_id=_mk_node_id())
        na1 = NamedArg(name="x", value=val1, span=sp, node_id=_mk_node_id())
        na2 = NamedArg(name="x", value=val2, span=sp, node_id=_mk_node_id())
        call = Call(
            callee=callee, args=(), named_args=(na1, na2), span=sp, node_id=_mk_node_id(),
        )
        block = Block(items=(fd, call), span=sp, node_id=_mk_node_id())
        prog = Program(body=block, span=sp, node_id=_mk_node_id())
        binding_ref = BindingRef(
            name="g", mutable=False, decl_span=sp, decl_node_id=fd_nid,
            kind=BinderKind.function_binding,
        )
        resolved = self._mk_resolved(
            prog,
            resolution={callee_nid: binding_ref},
            declared_functions={"g": fd},
        )
        with pytest.raises(AglTypeError, match="Duplicate argument"):
            check(resolved, default_capabilities())

    def test_duplicate_named_arg_in_constructor_rejected(self) -> None:
        # Exercises the duplicate named arg path in
        # ConstructorChecker._check_constructor_call (typecheck/constructors.py).
        # The parser rejects duplicate named args, so we construct the AST directly.
        from agm.agl.scope.symbols import ConstructorRef

        sp = mk_span()
        # Build a record type that has field 'x'.
        record_source = "record Box\n  x: int\nBox(x = 1)"
        prog_base = parse_program(record_source)
        res_base = resolve(prog_base)
        checked_base = check(res_base, default_capabilities())
        box_type = checked_base.type_env.get_type("Box")
        assert box_type is not None

        # Manually build a Call with duplicate named arg for 'x'.
        callee_nid = _mk_node_id()
        callee = VarRef(name="Box", span=sp, node_id=callee_nid)
        val1 = IntLit(value=1, span=sp, node_id=_mk_node_id())
        val2 = IntLit(value=2, span=sp, node_id=_mk_node_id())
        na1 = NamedArg(name="x", value=val1, span=sp, node_id=_mk_node_id())
        na2 = NamedArg(name="x", value=val2, span=sp, node_id=_mk_node_id())
        call_nid = _mk_node_id()
        call = Call(callee=callee, args=(), named_args=(na1, na2), span=sp, node_id=call_nid)
        block = Block(items=(call,), span=sp, node_id=_mk_node_id())
        prog_nid = _mk_node_id()
        prog = Program(body=block, span=sp, node_id=prog_nid)
        # Register a ConstructorRef for 'Box' and the callee VarRef.
        box_decl_node_id = prog_base.body.items[0].node_id
        ctor_ref = ConstructorRef(
            owner_name="Box",
            variant="Box",
            owner_decl_node_id=box_decl_node_id,
            type_params=(),
        )
        binding_ref = BindingRef(
            name="Box", mutable=False, decl_span=sp, decl_node_id=box_decl_node_id,
            kind=BinderKind.constructor_binding,
        )
        root = ScopeNode(node_id=prog_nid)
        from agm.agl.scope.symbols import ResolvedProgram as _RP

        resolved = _RP(
            program=prog,
            resolution={callee_nid: binding_ref},
            builtin_calls={},
            root_scope=root,
            constructor_refs={callee_nid: ctor_ref},
        )
        with pytest.raises(AglTypeError, match="[Dd]uplicate"):
            check(resolved, default_capabilities(), seed_env=checked_base.type_env)

    def test_type_arg_on_qualified_constructor_rejected(self) -> None:
        err = reject_type("enum Status\n  | Pass\n  | Fail\nStatus.Pass::[int]()\n()")
        assert "type argument" in str(err).lower()


# ---------------------------------------------------------------------------
# Fix 1: declared-name vs value-call dispatch is scope-aware
# ---------------------------------------------------------------------------


class TestCallDispatchScopeAware:
    """Verify that the declared-name/value-call dispatch uses BindingRef.kind,
    not the flat declared_functions name map, so that a let-bound function value
    that shadows a top-level def name is treated as a value call."""

    def test_shadow_def_with_let_fn_is_value_call(self) -> None:
        # 'classify' is a top-level def(a, b) -> text; inside wrap() a let
        # shadows it with a 1-param fn.  classify(5) in the inner scope must
        # type-check as a value call (1 positional arg, no named args).
        r = accept_type(
            "def classify(a: int, b: int) -> text = \"hi\"\n"
            "def wrap() -> int =\n"
            "  let classify: (int) -> int = fn(y: int) -> int => y\n"
            "  classify(5)\n"
            "wrap()"
        )
        assert r.resolved.program is not None

    def test_shadow_def_named_arg_on_let_fn_rejected(self) -> None:
        # Named arg on a let-bound fn (value call) must be rejected.
        err = reject_type(
            "def classify(a: int, b: int) -> text = \"hi\"\n"
            "def wrap() -> int =\n"
            "  let classify: (int) -> int = fn(y: int) -> int => y\n"
            "  classify(y = 5)\n"
            "wrap()"
        )
        assert "named" in str(err).lower() or "declared" in str(err).lower()

    def test_shadow_def_wrong_arity_on_let_fn_is_value_call_error(self) -> None:
        # The inner classify takes 1 param; calling with 2 is a value-call arity error.
        err = reject_type(
            "def classify(a: int, b: int) -> text = \"hi\"\n"
            "def wrap() -> int =\n"
            "  let classify: (int) -> int = fn(y: int) -> int => y\n"
            "  classify(5, 6)\n"
            "wrap()"
        )
        assert "arity" in str(err).lower() or "argument" in str(err).lower()

    def test_top_level_def_named_arg_still_works(self) -> None:
        # The top-level def must still accept named/defaulted args.
        r = accept_type(
            "def add(x: int, y: int = 0) -> int = x + y\n"
            "add(1, y = 2)"
        )
        assert r.resolved.program is not None

    def test_top_level_def_default_omission_still_works(self) -> None:
        # Omitting a defaulted argument at a top-level def call site still works.
        r = accept_type(
            "def add(x: int, y: int = 0) -> int = x + y\n"
            "add(3)"
        )
        assert r.resolved.program is not None

    def test_param_binding_callee_is_value_call(self) -> None:
        # A function parameter whose type is a function type must be treated as
        # a value call — positional only, named args rejected.
        r = accept_type(
            "def apply(f: (int) -> int, x: int) -> int = f(x)\n"
            "apply(fn(n: int) => n, 7)"
        )
        assert r.resolved.program is not None

    def test_param_binding_named_arg_rejected(self) -> None:
        # Named arg on a param-bound callee is rejected.
        err = reject_type(
            "def apply(f: (int) -> int, x: int) -> int = f(n = x)\n"
            "apply(fn(n: int) => n, 7)"
        )
        assert "named" in str(err).lower() or "declared" in str(err).lower()

    def test_field_access_callee_is_value_call(self) -> None:
        # A field that holds a function value must be treated as a value call.
        r = accept_type(
            "record Wrapper\n"
            "  fn_field: (int) -> int\n"
            "let w = Wrapper(fn_field = fn(x: int) -> int => x)\n"
            "w.fn_field(42)"
        )
        assert r.resolved.program is not None


# ---------------------------------------------------------------------------
# Fix 2: ask/exec reject unknown named args and extra positionals
# ---------------------------------------------------------------------------


class TestAskUnknownArgs:
    def test_ask_unknown_named_arg_rejected(self) -> None:
        err = reject_type('ask("Q", bogus = 1)')
        assert "ask" in str(err).lower() or "unknown" in str(err).lower()
        assert "bogus" in str(err)

    def test_ask_typo_named_arg_rejected(self) -> None:
        err = reject_type('ask("Q", strict_jsonn = true)')
        assert "ask" in str(err).lower() or "unknown" in str(err).lower()
        assert "strict_jsonn" in str(err)

    def test_ask_extra_positional_rejected(self) -> None:
        err = reject_type('ask("Q", "extra")')
        assert "ask" in str(err).lower() or "positional" in str(err).lower()

    def test_ask_valid_named_arg_combinations_still_accepted(self) -> None:
        # All four known named args together must be accepted.
        r = accept_type(
            'agent a\nlet n: int = ask("Q", agent = a, format = "json",'
            ' strict_json = true, on_parse_error = Abort())\nn'
        )
        assert r.resolved.program is not None


class TestExecUnknownArgs:
    def test_exec_unknown_named_arg_rejected(self) -> None:
        err = reject_type('exec("ls", bogus = 1)')
        assert "exec" in str(err).lower() or "unknown" in str(err).lower()
        assert "bogus" in str(err)

    def test_exec_agent_named_arg_rejected(self) -> None:
        # exec has no 'agent:' argument.
        err = reject_type('agent a\nexec("ls", agent = a)')
        assert "exec" in str(err).lower() or "unknown" in str(err).lower()

    def test_exec_extra_positional_rejected(self) -> None:
        err = reject_type('exec("ls", "extra")')
        assert "exec" in str(err).lower() or "positional" in str(err).lower()

    def test_exec_valid_named_arg_combinations_still_accepted(self) -> None:
        # format, strict_json, on_parse_error are valid for exec.
        r = accept_type(
            'let n: int = exec("ls", format = "json", strict_json = true,'
            ' on_parse_error = Abort())\nn'
        )
        assert r.resolved.program is not None


# ---------------------------------------------------------------------------
# Fix 3: structured exec rejects parse-shaping options + structured_exec flag
# ---------------------------------------------------------------------------


class TestExecStructured:
    def test_exec_no_annotation_sets_structured_exec(self) -> None:
        r = accept_type('exec("ls")')
        call = r.resolved.program.body.items[0]
        assert isinstance(call, Call)
        spec = r.contract_specs[call.node_id]
        assert spec.structured_exec is True
        assert spec.strict_json is None

    def test_exec_exec_result_annotation_sets_structured_exec(self) -> None:
        r = accept_type('let x: ExecResult = exec("ls")\nx')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        spec = r.contract_specs[decl.value.node_id]
        assert spec.structured_exec is True

    def test_exec_text_annotation_not_structured(self) -> None:
        r = accept_type('let x: text = exec("ls")\nx')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        spec = r.contract_specs[decl.value.node_id]
        assert spec.structured_exec is False

    def test_exec_structured_format_rejected(self) -> None:
        err = reject_type('exec("ls", format = "json")')
        assert "ExecResult" in str(err) or "format" in str(err).lower()

    def test_exec_structured_strict_json_rejected(self) -> None:
        err = reject_type('exec("ls", strict_json = true)')
        assert "ExecResult" in str(err) or "strict_json" in str(err).lower()

    def test_exec_structured_on_parse_error_rejected(self) -> None:
        err = reject_type('exec("ls", on_parse_error = Abort())')
        assert "ExecResult" in str(err) or "on_parse_error" in str(err).lower()

    def test_exec_parsed_form_has_no_structured_exec(self) -> None:
        r = accept_type('let n: int = exec("ls", format = "json")\nn')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        spec = r.contract_specs[decl.value.node_id]
        assert spec.structured_exec is False
        assert spec.codec_name == "json"


# ---------------------------------------------------------------------------
# Additional value-call error-case tests (reviewer-noted)
# ---------------------------------------------------------------------------


class TestValueCallErrors:
    def test_value_call_too_few_args(self) -> None:
        err = reject_type("let f = fn(x: int, y: int) -> int => x + y\nf(1)")
        assert "arity" in str(err).lower() or "argument" in str(err).lower()

    def test_value_call_too_many_args(self) -> None:
        err = reject_type("let f = fn(x: int) -> int => x\nf(1, 2)")
        assert "arity" in str(err).lower() or "argument" in str(err).lower()

    def test_value_call_positional_type_mismatch(self) -> None:
        # Passing text where int expected
        err = reject_type('let f = fn(x: int) -> int => x\nf("hello")')
        assert "mismatch" in str(err).lower() or "expected" in str(err).lower()

    def test_field_access_function_callee_value_call(self) -> None:
        # A record field holding a function value can be called as a value call.
        r = accept_type(
            "record Box\n"
            "  compute: (int) -> int\n"
            "let b = Box(compute = fn(n: int) -> int => n * 2)\n"
            "b.compute(5)"
        )
        assert r.resolved.program is not None

    def test_field_access_function_callee_named_arg_rejected(self) -> None:
        # Named arg on a field-access callee is rejected.
        err = reject_type(
            "record Box\n"
            "  compute: (int) -> int\n"
            "let b = Box(compute = fn(n: int) -> int => n * 2)\n"
            "b.compute(n = 5)"
        )
        assert "named" in str(err).lower() or "declared" in str(err).lower()

    def test_exec_function_target_rejected(self) -> None:
        # exec into a function/agent type is a static error.
        err = reject_type('let f: (int) -> int = exec("ls")\nf(1)')
        assert "function" in str(err).lower() or "agent" in str(err).lower()


# ---------------------------------------------------------------------------
# Generics: GenericTypeDef, ConstructorSignature, FunctionSignature.type_params
# ---------------------------------------------------------------------------


class TestGenericTypeDef:
    def test_register_and_get(self) -> None:
        env = TypeEnvironment()
        template = RecordType("Box", {"value": TypeVarType("T")})
        gdef = GenericTypeDef(kind="record", type_params=("T",), template=template)
        env.register_generic_type("Box", gdef)
        result = env.get_generic_type("Box")
        assert result == gdef

    def test_get_unknown_returns_none(self) -> None:
        env = TypeEnvironment()
        assert env.get_generic_type("Unknown") is None

    def test_instantiate_nominal_record(self) -> None:
        env = TypeEnvironment()
        template = RecordType("Box", {"value": TypeVarType("T")})
        gdef = GenericTypeDef(kind="record", type_params=("T",), template=template)
        env.register_generic_type("Box", gdef)
        result = env.instantiate_nominal("Box", (IntType(),))
        assert isinstance(result, RecordType)
        assert result.name == "Box"
        assert result.type_args == (IntType(),)
        assert result.fields["value"] == IntType()

    def test_instantiate_nominal_enum(self) -> None:
        env = TypeEnvironment()
        template = EnumType("Option", {"Some": {"value": TypeVarType("T")}, "None": {}})
        gdef = GenericTypeDef(kind="enum", type_params=("T",), template=template)
        env.register_generic_type("Option", gdef)
        result = env.instantiate_nominal("Option", (TextType(),))
        assert isinstance(result, EnumType)
        assert result.type_args == (TextType(),)
        assert result.variants["Some"]["value"] == TextType()
        assert result.variants["None"] == {}

    def test_instantiate_arity_mismatch(self) -> None:
        env = TypeEnvironment()
        template = RecordType("Box", {"value": TypeVarType("T")})
        gdef = GenericTypeDef(kind="record", type_params=("T",), template=template)
        env.register_generic_type("Box", gdef)
        with pytest.raises(AglTypeError, match="1"):
            env.instantiate_nominal("Box", (IntType(), TextType()))

    def test_instantiate_zero_field_record(self) -> None:
        env = TypeEnvironment()
        template = RecordType("Marker", {})
        gdef = GenericTypeDef(kind="record", type_params=(), template=template)
        env.register_generic_type("Marker", gdef)
        result = env.instantiate_nominal("Marker", ())
        assert isinstance(result, RecordType)
        assert result.fields == {}


class TestConstructorSignature:
    def test_register_and_get(self) -> None:
        env = TypeEnvironment()
        template_result = RecordType("Box", {"value": TypeVarType("T")})
        sig = ConstructorSignature(
            owner_name="Box",
            variant=None,
            field_names=("value",),
            field_templates=(TypeVarType("T"),),
            result_template=template_result,
            type_params=("T",),
        )
        env.register_constructor_signature(sig)
        got = env.get_constructor_signature("Box", None)
        assert got == sig

    def test_enum_variant_signature(self) -> None:
        env = TypeEnvironment()
        template_result = EnumType("Option", {"Some": {"value": TypeVarType("T")}, "None": {}})
        sig = ConstructorSignature(
            owner_name="Option",
            variant="Some",
            field_names=("value",),
            field_templates=(TypeVarType("T"),),
            result_template=template_result,
            type_params=("T",),
        )
        env.register_constructor_signature(sig)
        got = env.get_constructor_signature("Option", "Some")
        assert got is not None
        assert got.variant == "Some"

    def test_get_nonexistent_returns_none(self) -> None:
        env = TypeEnvironment()
        assert env.get_constructor_signature("Foo", None) is None

    def test_nullary_variant_empty_fields(self) -> None:
        env = TypeEnvironment()
        template_result = EnumType("Option", {"Some": {}, "None": {}})
        sig = ConstructorSignature(
            owner_name="Option",
            variant="None",
            field_names=(),
            field_templates=(),
            result_template=template_result,
            type_params=("T",),
        )
        env.register_constructor_signature(sig)
        got = env.get_constructor_signature("Option", "None")
        assert got is not None
        assert got.field_names == ()


class TestFunctionSignatureTypeParams:
    def test_default_type_params_empty(self) -> None:
        sig = FunctionSignature(params=(), result=IntType())
        assert sig.type_params == ()

    def test_with_type_params(self) -> None:
        sig = FunctionSignature(params=(), result=TypeVarType("T"), type_params=("T",))
        assert sig.type_params == ("T",)


class TestResolveTypeExprTypeVars:
    def test_qualified_applied_type_requires_graph_context(self) -> None:
        from agm.agl.syntax.types import AppliedT, Qualifier

        sp = mk_span()
        expr = AppliedT(
            name="Box",
            args=(IntT(span=sp, node_id=1),),
            span=sp,
            node_id=2,
            module_qualifier=Qualifier(segments=("lib",), span=sp, node_id=3),
        )
        with pytest.raises(AglTypeError, match="outside of a module graph"):
            TypeEnvironment().resolve_type_expr(expr)

    def test_self_qualified_unknown_applied_type_rejected(self) -> None:
        err = reject_type("let x: ::Missing[int] = null\nx")
        assert "Unknown type 'Missing'" in str(err)

    def test_open_imported_generic_lookup_edge_cases(self) -> None:
        from agm.agl.modules.ids import ModuleId
        from agm.agl.scope.imports import ImportEnv
        from agm.agl.semantics.types import TypeVarType
        from agm.agl.typecheck.env import GenericTypeDef

        lib_a = ModuleId.from_dotted("a")
        lib_b = ModuleId.from_dotted("b")
        gdef = GenericTypeDef(
            kind="record",
            type_params=("T",),
            template=RecordType("Box", {"value": TypeVarType("T")}),
        )
        assert TypeEnvironment().get_open_imported_generic_type("Box") is None
        env = TypeEnvironment(
            import_env=ImportEnv(
                unqualified={
                    "Point": frozenset({(lib_a, "Point")}),
                    "Box": frozenset({(lib_a, "Box"), (lib_b, "Box")}),
                },
                qualified={},
            ),
            graph_generic_table={(lib_a, "Box"): gdef, (lib_b, "Box"): gdef},
        )
        assert env.get_open_imported_generic_type("Point") is None
        assert env.get_open_imported_generic_type("Box") is None

    def test_name_in_type_vars_resolves_to_typevar(self) -> None:
        from agm.agl.syntax.types import NameT
        env = TypeEnvironment()
        sp = mk_span()
        result = env.resolve_type_expr(
            NameT(name="T", span=sp, node_id=1), type_vars=frozenset({"T"})
        )
        assert result == TypeVarType("T")

    def test_registered_type_resolves_with_type_vars_ignored(self) -> None:
        from agm.agl.syntax.types import NameT
        env = TypeEnvironment()
        sp = mk_span()
        env.register_type("MyRec", RecordType("MyRec", {}))
        result = env.resolve_type_expr(
            NameT(name="MyRec", span=sp, node_id=1), type_vars=frozenset({"T"})
        )
        assert isinstance(result, RecordType)
        assert result.name == "MyRec"

    def test_applied_t_resolves_generic_type(self) -> None:
        from agm.agl.syntax.types import AppliedT
        env = TypeEnvironment()
        sp = mk_span()
        template = RecordType("Box", {"value": TypeVarType("T")})
        gdef = GenericTypeDef(kind="record", type_params=("T",), template=template)
        env.register_generic_type("Box", gdef)
        result = env.resolve_type_expr(
            AppliedT(name="Box", args=(IntT(span=sp, node_id=2),), span=sp, node_id=3),
            type_vars=frozenset(),
        )
        assert isinstance(result, RecordType)
        assert result.type_args == (IntType(),)
        assert result.fields["value"] == IntType()

    def test_applied_t_arity_mismatch_raises(self) -> None:
        from agm.agl.syntax.types import AppliedT
        env = TypeEnvironment()
        sp = mk_span()
        template = RecordType("Box", {"value": TypeVarType("T")})
        gdef = GenericTypeDef(kind="record", type_params=("T",), template=template)
        env.register_generic_type("Box", gdef)
        with pytest.raises(AglTypeError):
            env.resolve_type_expr(
                AppliedT(
                    name="Box",
                    args=(IntT(span=sp, node_id=1), IntT(span=sp, node_id=2)),
                    span=sp,
                    node_id=3,
                ),
            )

    def test_bare_generic_name_rejected(self) -> None:
        from agm.agl.syntax.types import NameT
        env = TypeEnvironment()
        sp = mk_span()
        template = RecordType("Box", {"value": TypeVarType("T")})
        gdef = GenericTypeDef(kind="record", type_params=("T",), template=template)
        env.register_generic_type("Box", gdef)
        with pytest.raises(AglTypeError, match="1"):
            env.resolve_type_expr(NameT(name="Box", span=sp, node_id=1))

    def test_applied_t_non_generic_raises(self) -> None:
        from agm.agl.syntax.types import AppliedT
        env = TypeEnvironment()
        sp = mk_span()
        with pytest.raises(AglTypeError):
            env.resolve_type_expr(
                AppliedT(name="Unknown", args=(IntT(span=sp, node_id=1),), span=sp, node_id=2),
            )

    def test_parameterized_alias_applied(self) -> None:
        from agm.agl.syntax.types import AppliedT, NameT
        from agm.agl.syntax.types import ListT as _ListT
        env = TypeEnvironment()
        sp = mk_span()
        env.register_alias(
            "Wrapper",
            _ListT(elem=NameT(name="T", span=sp, node_id=10), span=sp, node_id=11),
            type_params=("T",),
        )
        result = env.resolve_type_expr(
            AppliedT(name="Wrapper", args=(IntT(span=sp, node_id=1),), span=sp, node_id=2),
        )
        assert result == ListType(IntType())

    def test_bare_parameterized_alias_rejected(self) -> None:
        from agm.agl.syntax.types import ListT as _ListT
        from agm.agl.syntax.types import NameT
        env = TypeEnvironment()
        sp = mk_span()
        env.register_alias(
            "Wrapper",
            _ListT(elem=NameT(name="T", span=sp, node_id=10), span=sp, node_id=11),
            type_params=("T",),
        )
        with pytest.raises(AglTypeError, match="1"):
            env.resolve_type_expr(NameT(name="Wrapper", span=sp, node_id=2))

    def test_seed_from_copies_generic_types(self) -> None:
        env1 = TypeEnvironment()
        template = RecordType("Box", {"value": TypeVarType("T")})
        gdef = GenericTypeDef(kind="record", type_params=("T",), template=template)
        env1.register_generic_type("Box", gdef)
        env2 = TypeEnvironment()
        env2.seed_from(env1)
        assert env2.get_generic_type("Box") == gdef

    def test_seed_from_copies_constructor_sigs(self) -> None:
        env1 = TypeEnvironment()
        sig = ConstructorSignature(
            owner_name="Box",
            variant=None,
            field_names=("value",),
            field_templates=(TypeVarType("T"),),
            result_template=RecordType("Box", {"value": TypeVarType("T")}),
            type_params=("T",),
        )
        env1.register_constructor_signature(sig)
        env2 = TypeEnvironment()
        env2.seed_from(env1)
        assert env2.get_constructor_signature("Box", None) == sig

    def test_seed_from_copies_alias_type_params(self) -> None:
        from agm.agl.syntax.types import ListT as _ListT
        from agm.agl.syntax.types import NameT
        env1 = TypeEnvironment()
        sp = mk_span()
        env1.register_alias(
            "Wrapper",
            _ListT(elem=NameT(name="T", span=sp, node_id=1), span=sp, node_id=2),
            type_params=("T",),
        )
        env2 = TypeEnvironment()
        env2.seed_from(env1)
        assert env2.get_alias_type_params("Wrapper") == ("T",)

    def test_instantiate_nominal_unknown_raises(self) -> None:
        env = TypeEnvironment()
        with pytest.raises(AglTypeError, match="Unknown generic type"):
            env.instantiate_nominal("NotRegistered", ())

    def test_applied_t_alias_arity_mismatch_raises(self) -> None:
        from agm.agl.syntax.types import AppliedT, NameT
        from agm.agl.syntax.types import ListT as _ListT
        env = TypeEnvironment()
        sp = mk_span()
        env.register_alias(
            "Wrapper",
            _ListT(elem=NameT(name="T", span=sp, node_id=10), span=sp, node_id=11),
            type_params=("T",),
        )
        with pytest.raises(AglTypeError, match="requires 1"):
            env.resolve_type_expr(
                AppliedT(
                    name="Wrapper",
                    args=(IntT(span=sp, node_id=1), IntT(span=sp, node_id=2)),
                    span=sp,
                    node_id=3,
                ),
            )

    def test_applied_t_on_non_generic_registered_type_raises(self) -> None:
        from agm.agl.syntax.types import AppliedT
        env = TypeEnvironment()
        sp = mk_span()
        env.register_type("Plain", RecordType("Plain", {}))
        with pytest.raises(AglTypeError, match="does not take type arguments"):
            env.resolve_type_expr(
                AppliedT(name="Plain", args=(IntT(span=sp, node_id=1),), span=sp, node_id=2),
            )

    def test_applied_t_unknown_nested_arg_span_is_arg_span(self) -> None:
        # Error span for an unknown type inside a type argument must point at
        # the argument node, not at the outer AppliedT span.
        from agm.agl.syntax.types import AppliedT, NameT
        env = TypeEnvironment()
        outer_sp = mk_span(line=1, col=1)
        arg_sp = mk_span(line=5, col=10)
        template = RecordType("Box", {"value": TypeVarType("T")})
        gdef = GenericTypeDef(kind="record", type_params=("T",), template=template)
        env.register_generic_type("Box", gdef)
        with pytest.raises(AglTypeError) as exc_info:
            env.resolve_type_expr(
                AppliedT(
                    name="Box",
                    args=(NameT(name="NoSuchType", span=arg_sp, node_id=1),),
                    span=outer_sp,
                    node_id=2,
                ),
            )
        assert exc_info.value.span == arg_sp


# ---------------------------------------------------------------------------
# TypeVarType: derive_schema raises TypeError (coverage for schema.py)
# ---------------------------------------------------------------------------


class TestTypeVarTypeSchema:
    def test_typevar_type_not_wire_serialisable(self) -> None:
        from agm.agl.semantics.type_table import create_seeded_type_table
        from agm.agl.type_schema import derive_schema
        with pytest.raises(TypeError, match="TypeVarType"):
            derive_schema(TypeVarType("T"), create_seeded_type_table())


# ---------------------------------------------------------------------------
# Generic def checking, type-argument solver, parametricity gates
# ---------------------------------------------------------------------------


class TestGenerics:
    """Tests for generic def type-checking, inference,  parametricity,
     target guard, and  generic-def-as-value instantiation."""

    # ------------------------------------------------------------------
    # Generic def: body checking
    # ------------------------------------------------------------------

    def test_generic_id_def_accepted(self) -> None:
        accept_type("def id[T](x: T) -> T = x\nid(1)")

    def test_generic_const_def_accepted(self) -> None:
        accept_type("def const[A, B](a: A, b: B) -> A = a\nconst(1, true)")

    def test_generic_first_container_index_allowed(self) -> None:
        # list[T] indexing yields T — indexing on a container of T is fine
        accept_type("def first[T](xs: list[T]) -> T = xs[0]\nfirst([1, 2, 3])")

    def test_generic_wrap_def_accepted(self) -> None:
        accept_type("def wrap[T](x: T) -> list[T] = [x]\nwrap(1)")

    # ------------------------------------------------------------------
    # Type argument inference
    # ------------------------------------------------------------------

    def test_inference_arg_driven(self) -> None:
        r = accept_type("def id[T](x: T) -> T = x\nlet n = id(1)\nn")
        decl = r.resolved.program.body.items[1]
        assert isinstance(decl, LetDecl)
        assert r.type_env.get_binding_type(decl.node_id) == IntType()

    def test_inference_result_only_from_expected(self) -> None:
        # T only appears in the result; context provides the binding
        r = accept_type("def empty[T]() -> list[T] = []\nlet xs: list[int] = empty()\nxs")
        decl = r.resolved.program.body.items[1]
        assert isinstance(decl, LetDecl)
        assert r.type_env.get_binding_type(decl.node_id) == ListType(elem=IntType())

    def test_inference_context_doesnt_override_arg(self) -> None:
        # let x: decimal = id(1) infers T=int, then coerces int → decimal
        r = accept_type("def id[T](x: T) -> T = x\nlet x: decimal = id(1)\nx")
        decl = r.resolved.program.body.items[1]
        assert isinstance(decl, LetDecl)
        assert r.type_env.get_binding_type(decl.node_id) == DecimalType()

    def test_explicit_type_args_single(self) -> None:
        r = accept_type("def id[T](x: T) -> T = x\nid::[int](1)")
        assert r.resolved.program is not None

    def test_explicit_type_args_multi(self) -> None:
        r = accept_type("def const[A, B](a: A, b: B) -> A = a\nconst::[int, text](1, \"x\")")
        assert r.resolved.program is not None

    def test_explicit_type_args_function_value(self) -> None:
        r = accept_type("def f[T](x: T) -> T = x\nf::[int]")
        prog = r.resolved.program
        expr = prog.body.items[1]
        assert isinstance(expr, TypeApply)
        assert r.node_types[expr.node_id] == FunctionType(params=(IntType(),), result=IntType())

    def test_explicit_type_args_function_value_can_be_bound(self) -> None:
        r = accept_type("def f[T](x: T) -> T = x\nlet g = f::[int]\ng(1)")
        assert r.resolved.program is not None

    def test_explicit_type_args_function_value_arity_error(self) -> None:
        err = reject_type("def f[T](x: T) -> T = x\nf::[int, text]")
        assert "type argument" in str(err).lower()

    def test_explicit_type_args_function_value_requires_name(self) -> None:
        err = reject_type("(fn(x: int) -> int => x)::[int]")
        assert "generic function value" in str(err)

    def test_explicit_type_args_function_value_requires_function_binding(self) -> None:
        err = reject_type("let x = 1\nx::[int]")
        assert "generic function value" in str(err)

    def test_explicit_type_args_function_value_requires_generic_function(self) -> None:
        err = reject_type("def f(x: int) -> int = x\nf::[int]")
        assert "not a generic function" in str(err)

    def test_explicit_type_args_function_value_requires_known_signature(self) -> None:
        err = reject_type("f::[int]\ndef f[T](x: T) = x")
        assert "return type" in str(err)
        assert "annotation" in str(err)

    def test_explicit_type_args_arity_error(self) -> None:
        err = reject_type("def id[T](x: T) -> T = x\nid::[int, text](1)")
        assert "type argument" in str(err).lower() or "arity" in str(err).lower()

    def test_inconsistent_binding_error(self) -> None:
        # pair[T](a: T, b: T) — T=int from first arg, T must be text from second
        err = reject_type('def pair[T](a: T, b: T) -> T = a\npair(1, "x")')
        assert "inconsistent" in str(err).lower() or "type argument" in str(err).lower()

    def test_uninferable_variable_error(self) -> None:
        # Result-only type var with no expected context
        err = reject_type("def empty[T]() -> list[T] = []\nempty()")
        assert (
            "infer" in str(err).lower()
            or "type argument" in str(err).lower()
            or "supply" in str(err).lower()
        )

    # ------------------------------------------------------------------
    # Strict parametricity — reject operations on bare TypeVarType
    # ------------------------------------------------------------------

    def test_d2_equality_on_T_rejected(self) -> None:
        err = reject_type("def eq[T](a: T, b: T) -> bool = a == b")
        assert (
            "type variable" in str(err).lower()
            or "abstract" in str(err).lower()
            or "not permitted" in str(err).lower()
        )

    def test_d2_ordering_on_T_rejected(self) -> None:
        err = reject_type("def lt[T](a: T, b: T) -> bool = a < b")
        assert "type variable" in str(err).lower() or "abstract" in str(err).lower()

    def test_d2_add_on_T_rejected(self) -> None:
        err = reject_type("def add[T](a: T, b: T) -> T = a + b")
        assert (
            "type variable" in str(err).lower()
            or "abstract" in str(err).lower()
            or "not permitted" in str(err).lower()
        )

    def test_d2_sub_on_T_rejected(self) -> None:
        err = reject_type("def sub[T](a: T, b: T) -> T = a - b")
        assert "type variable" in str(err).lower() or "abstract" in str(err).lower()

    def test_d2_mul_on_T_rejected(self) -> None:
        err = reject_type("def mul[T](a: T, b: T) -> T = a * b")
        assert "type variable" in str(err).lower() or "abstract" in str(err).lower()

    def test_d2_div_on_T_rejected(self) -> None:
        err = reject_type("def div[T](a: T, b: T) -> T = a / b")
        assert "type variable" in str(err).lower() or "abstract" in str(err).lower()

    def test_d2_unary_neg_on_T_rejected(self) -> None:
        err = reject_type("def neg[T](x: T) -> T = -x")
        assert "type variable" in str(err).lower() or "abstract" in str(err).lower()

    def test_d2_print_on_T_accepted(self) -> None:
        accept_type("def show[T](x: T) -> unit = print(x)")

    def test_d2_interpolation_on_T_accepted(self) -> None:
        accept_type('def show[T](x: T) -> text = "${x}"')

    def test_d2_field_access_on_T_rejected(self) -> None:
        err = reject_type("def get[T](x: T) -> int = x.field")
        assert (
            "type variable" in str(err).lower()
            or "abstract" in str(err).lower()
            or "field" in str(err).lower()
        )

    def test_d2_index_on_T_rejected(self) -> None:
        err = reject_type("def get[T](x: T) -> int = x[0]")
        assert (
            "type variable" in str(err).lower()
            or "abstract" in str(err).lower()
            or "indexable" in str(err).lower()
        )

    def test_d2_is_test_on_T_rejected(self) -> None:
        err = reject_type("enum E\n  | A\ndef check[T](x: T) -> bool = x is E.A")
        assert "type variable" in str(err).lower() or "abstract" in str(err).lower()

    def test_d2_in_op_bare_T_rejected(self) -> None:
        # T `in` list[T] — left operand is a bare TypeVarType
        err = reject_type("def contains[T](x: T, xs: list[T]) -> bool = x in xs")
        assert (
            "type variable" in str(err).lower()
            or "abstract" in str(err).lower()
            or "not permitted" in str(err).lower()
        )

    def test_d2_container_of_T_index_allowed(self) -> None:
        # xs: list[T]; xs[0] yields T — container index is fine
        accept_type("def first[T](xs: list[T]) -> T = xs[0]\nfirst([1])")

    # ------------------------------------------------------------------
    # Generic def used as a value
    # ------------------------------------------------------------------

    def test_d5_generic_def_as_value_with_expected(self) -> None:
        r = accept_type("def id[T](x: T) -> T = x\nlet f: (int) -> int = id\nf(1)")
        assert r.resolved.program is not None

    def test_d5_generic_def_as_value_no_expected_errors(self) -> None:
        err = reject_type("def id[T](x: T) -> T = x\nlet f = id\nf")
        assert (
            "infer" in str(err).lower()
            or "type argument" in str(err).lower()
            or "annotate" in str(err).lower()
        )

    def test_d5_generic_def_still_callable(self) -> None:
        accept_type("def id[T](x: T) -> T = x\nid(42)")

    # ------------------------------------------------------------------
    # Agent/exec target may not contain a type variable
    # ------------------------------------------------------------------

    def test_d3_ask_with_type_var_target_rejected(self) -> None:
        err = reject_type('def fetch[T](p: text) -> T = ask::[T](p)')
        assert "type variable" in str(err).lower() or "cannot" in str(err).lower()

    def test_d3_ask_explicit_concrete_target_accepted(self) -> None:
        r = accept_type('let n: int = ask::[int]("Q")\nn')
        assert r.resolved.program is not None

    def test_d3_ask_explicit_too_many_args_error(self) -> None:
        err = reject_type('ask::[int, text]("Q")')
        assert "type argument" in str(err).lower()

    def test_d3_exec_with_type_var_rejected(self) -> None:
        err = reject_type('def run[T](cmd: text) -> T = exec::[T](cmd)')
        assert "type variable" in str(err).lower() or "cannot" in str(err).lower()

    def test_d3_exec_explicit_concrete_target_accepted(self) -> None:
        r = accept_type('let x: text = exec::[text]("ls")\nx')
        assert r.resolved.program is not None

    def test_d3_exec_too_many_type_args_error(self) -> None:
        err = reject_type('exec::[text, int]("ls")')
        assert "type argument" in str(err).lower()

    def test_d3_ask_request_with_type_var_rejected(self) -> None:
        err = reject_type('def req[T](p: text) -> AgentRequest = ask-request::[T](p)')
        assert "type variable" in str(err).lower() or "cannot" in str(err).lower()

    def test_d3_ask_with_list_of_T_rejected(self) -> None:
        # list[T] as target also contains a type var
        err = reject_type('def fetch[T](p: text) -> list[T] = ask::[list[T]](p)')
        assert "type variable" in str(err).lower() or "cannot" in str(err).lower()

    def test_d3_ask_contextual_type_var_target_rejected(self) -> None:
        # Regression: a bare type-var target inferred from the generic def's
        # return type (no explicit ::[…]) must give the clean  error, not a
        # codec-selection error.
        err = reject_type('def fetch[T](p: text) -> T = ask(p)')
        assert "type variable" in str(err).lower()

    def test_d3_ask_contextual_composite_type_var_target_rejected(self) -> None:
        # Regression: a composite contextual target (list[T]) previously crashed
        # in schema generation; it must now be a clean static error.
        err = reject_type('def fetch[T](p: text) -> list[T] = ask(p)')
        assert "type variable" in str(err).lower()

    def test_d3_exec_contextual_type_var_target_rejected(self) -> None:
        err = reject_type('def run[T](cmd: text) -> T = exec(cmd)')
        assert "type variable" in str(err).lower()

    def test_d3_ask_contextual_concrete_generic_instance_accepted(self) -> None:
        # A concrete generic instance (no free type var) is a valid ask target.
        r = accept_type(
            "enum Option[T]\n  | none\n  | some(value: T)\n"
            'let x: Option[int] = ask("Q")\nx'
        )
        assert r.resolved.program is not None

    def test_d3_non_type_var_cases_unaffected(self) -> None:
        # Concrete targets (non-generic context) are unaffected
        accept_type('let n: int = ask("Q")\nn')
        accept_type('let x: text = exec("ls")\nx')

    # ------------------------------------------------------------------
    # right-operand TypeVarType branches (separate from left)
    # ------------------------------------------------------------------

    def test_d2_right_eq_T_rejected(self) -> None:
        # left is concrete, right is TypeVarType
        err = reject_type("def f[T](x: T) -> bool = 1 == x")
        assert "type variable" in str(err).lower() or "not permitted" in str(err).lower()

    def test_d2_right_ordering_T_rejected(self) -> None:
        err = reject_type("def f[T](x: T) -> bool = 1 < x")
        assert "type variable" in str(err).lower() or "abstract" in str(err).lower()

    def test_d2_right_add_T_rejected(self) -> None:
        err = reject_type("def f[T](x: T) -> T = 1 + x")
        assert "type variable" in str(err).lower() or "not permitted" in str(err).lower()

    def test_d2_right_sub_T_rejected(self) -> None:
        err = reject_type("def f[T](x: T) -> T = 1 - x")
        assert "type variable" in str(err).lower() or "abstract" in str(err).lower()

    def test_d2_right_mul_T_rejected(self) -> None:
        err = reject_type("def f[T](x: T) -> T = 1 * x")
        assert "type variable" in str(err).lower() or "abstract" in str(err).lower()

    def test_d2_right_div_T_rejected(self) -> None:
        err = reject_type("def f[T](x: T) -> T = 1 / x")
        assert "type variable" in str(err).lower() or "abstract" in str(err).lower()

    def test_d2_right_in_T_rejected(self) -> None:
        # right operand is TypeVarType in an 'in' operation
        err = reject_type('def f[T](x: T) -> bool = "a" in x')
        assert "type variable" in str(err).lower() or "abstract" in str(err).lower()

    # ------------------------------------------------------------------
    # cannot infer type arg for generic-as-value from context
    # ------------------------------------------------------------------

    def test_d5_uninferable_from_wrong_arity_context(self) -> None:
        # id is (T) -> T but context is () -> int (arity mismatch) — T stays unsolved
        err = reject_type("def id[T](x: T) -> T = x\nlet f: () -> int = id\nf()")
        assert (
            "infer" in str(err).lower()
            or "type argument" in str(err).lower()
            or "annotate" in str(err).lower()
        )

    # ------------------------------------------------------------------
    # _match structural: DictType and FunctionType recursion
    # ------------------------------------------------------------------

    def test_match_dict_T_inferred(self) -> None:
        # dict[text, T] param: matching infers T from a dict[text, int] value
        r = accept_type(
            'def first_val[T](d: dict[text, T]) -> T = d["k"]\nfirst_val({"k": 1})'
        )
        assert r.resolved.program is not None

    def test_match_function_T_inferred(self) -> None:
        # (T) -> T param: matching against a concrete function infers T
        r = accept_type(
            "def apply[T](f: (T) -> T, x: T) -> T = f(x)\n"
            "def inc(n: int) -> int = n + 1\n"
            "apply(inc, 1)"
        )
        assert r.resolved.program is not None

    # ------------------------------------------------------------------
    # _match_unsolved: DictType and FunctionType structural recursion
    # ------------------------------------------------------------------

    def test_match_unsolved_dict_T_from_expected(self) -> None:
        # empty[T]() -> dict[text, T]: T inferred from expected dict[text, text]
        r = accept_type(
            "def empty_dict[T]() -> dict[text, T] = {}\n"
            "let d: dict[text, text] = empty_dict()\nd"
        )
        assert r.resolved.program is not None

    def test_match_unsolved_non_matching_result_type_ignored(self) -> None:
        # When sig.result is a concrete type, _match_unsolved is a no-op —
        # the type mismatch is caught by the final assignability check instead.
        err = reject_type("def f[T](x: T) -> text = \"hi\"\nlet n: int = f(1)\nn")
        assert (
            "text" in str(err).lower()
            or "int" in str(err).lower()
            or "assign" in str(err).lower()
        )

    # ------------------------------------------------------------------
    # Named args in generic calls (inference + substitution paths)
    # ------------------------------------------------------------------

    def test_generic_named_arg_call_accepted(self) -> None:
        r = accept_type("def f[T](x: T, y: T) -> T = x\nf(x = 1, y = 2)")
        assert r.resolved.program is not None

    def test_generic_named_arg_call_error_unknown(self) -> None:
        # T is inferred from positional arg; then named arg 'z' is unknown
        err = reject_type("def f[T](x: T, y: T) -> T = x\nf(1, z = 2)")
        assert "unknown" in str(err).lower() or "parameter" in str(err).lower()

    def test_generic_named_arg_positional_and_named_duplicate(self) -> None:
        err = reject_type("def f[T](x: T) -> T = x\nf(1, x = 1)")
        assert (
            "positional" in str(err).lower()
            or "parameter" in str(err).lower()
            or "name" in str(err).lower()
            or "both" in str(err).lower()
            or "duplicate" in str(err).lower()
        )

    # ------------------------------------------------------------------
    # Missing required arg and too many args in generic calls
    # ------------------------------------------------------------------

    def test_generic_too_many_positional_args_error(self) -> None:
        err = reject_type("def f[T](x: T) -> T = x\nf(1, 2)")
        assert (
            "too many" in str(err).lower()
            or "argument" in str(err).lower()
            or "parameter" in str(err).lower()
        )

    def test_generic_missing_required_arg_error(self) -> None:
        err = reject_type("def f[T](x: T, y: T) -> T = x\nf(1)")
        assert "missing" in str(err).lower() or "required" in str(err).lower()

    def test_generic_call_with_defaulted_param_omitted(self) -> None:
        # Covers the bound_expr-is-None branch in the new inference loop
        # (a defaulted param is omitted at the call site).
        r = accept_type("def f[T](x: T, y: int = 5) -> T = x\nf(\"hello\")")
        assert r.resolved.program is not None

    # ------------------------------------------------------------------
    # Non-generic function rejects explicit type args
    # ------------------------------------------------------------------

    def test_nongeneric_explicit_type_arg_rejected(self) -> None:
        err = reject_type("def f(x: int) -> int = x\nf::[int](1)")
        assert (
            "not a generic" in str(err).lower()
            or "type argument" in str(err).lower()
            or "generic" in str(err).lower()
        )

    # ------------------------------------------------------------------
    # _match: non-TypeVar/List/Dict/Function template (1300->exit) and
    # FunctionType arity mismatch (1301->exit)
    # ------------------------------------------------------------------

    def test_match_concrete_param_in_generic_def(self) -> None:
        # Concrete param (int) → _match(IntType(), IntType(), ...) falls off
        # the elif chain (1300->exit); T inferred from second param
        r = accept_type('def f[T](x: int, y: T) -> T = y\nf(1, "hi")')
        assert r.resolved.program is not None

    def test_match_function_arity_mismatch_falls_off(self) -> None:
        # Template (T)->T matched against (int,int)->int: arity mismatch (1301->exit)
        # _match silently gives up; T is inferred from x: T = 1 = int;
        # then (int,int)->int is checked against (int)->int → type mismatch
        err = reject_type(
            "def apply[T](f: (T) -> T, x: T) -> T = f(x)\n"
            "def two(a: int, b: int) -> int = a + b\n"
            "apply(two, 1)"
        )
        assert (
            "infer" in str(err).lower()
            or "type argument" in str(err).lower()
            or "mismatch" in str(err).lower()
            or "assign" in str(err).lower()
        )

# ---------------------------------------------------------------------------
# Generic record/enum type declarations
# ---------------------------------------------------------------------------


class TestGenericTypeDecl:
    """Tests for generic record and enum type declarations."""

    def test_generic_record_accepted(self) -> None:
        r = accept_type(
            "record Box[T]\n"
            "  value: T\n"
            "let b: Box[int] = Box(value = 42)\nb"
        )
        assert r.resolved.program is not None

    def test_generic_enum_accepted(self) -> None:
        r = accept_type(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "let x: Option[int] = some(value = 1)\nx"
        )
        assert r.resolved.program is not None

    def test_generic_record_two_type_params(self) -> None:
        r = accept_type(
            "record Pair[A, B]\n"
            "  first: A\n"
            "  second: B\n"
            'let p: Pair[int, text] = Pair(first = 1, second = "hi")\np'
        )
        assert r.resolved.program is not None

    def test_generic_enum_two_type_params(self) -> None:
        r = accept_type(
            "enum Either[L, R]\n"
            "  | left(value: L)\n"
            "  | right(value: R)\n"
            "let e: Either[int, text] = left(value = 1)\ne"
        )
        assert r.resolved.program is not None

    def test_generic_record_with_concrete_field(self) -> None:
        r = accept_type(
            "record Tagged[T]\n"
            "  label: text\n"
            "  value: T\n"
            'let t: Tagged[int] = Tagged(label = "n", value = 5)\nt'
        )
        assert r.resolved.program is not None

    def test_generic_type_registers_in_env(self) -> None:
        r = accept_type(
            "record Box[T]\n"
            "  value: T\n"
            "let b: Box[int] = Box(value = 1)\nb"
        )
        gdef = r.type_env.get_generic_type("Box")
        assert gdef is not None
        assert gdef.type_params == ("T",)

    def test_bare_generic_name_without_args_rejected(self) -> None:
        err = reject_type(
            "record Box[T]\n"
            "  value: T\n"
            "let b: Box = Box(value = 1)\nb"
        )
        assert "type argument" in str(err).lower()

    def test_generic_record_duplicate_field_rejected(self) -> None:
        err = reject_type(
            "record Bad[T]\n"
            "  x: T\n"
            "  x: int\n"
            "Bad(x = 1)"
        )
        assert "duplicate" in str(err).lower() or "field" in str(err).lower()

    def test_generic_enum_duplicate_variant_rejected(self) -> None:
        err = reject_type(
            "enum Bad[T]\n"
            "  | Foo\n"
            "  | Foo\n"
            "Foo()"
        )
        assert "duplicate" in str(err).lower() or "variant" in str(err).lower()


class TestGenericConstructorInference:
    """Tests for type-argument inference on generic constructors."""

    def test_record_constructor_inferred_from_field(self) -> None:
        r = accept_type(
            "record Box[T]\n"
            "  value: T\n"
            "Box(value = 42)"
        )
        assert r.resolved.program is not None

    def test_record_constructor_inferred_from_annotation(self) -> None:
        r = accept_type(
            "record Box[T]\n"
            "  value: T\n"
            "let b: Box[int] = Box(value = 42)\nb"
        )
        assert r.resolved.program is not None

    def test_enum_payload_variant_inferred_from_field(self) -> None:
        r = accept_type(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "some(value = 1)"
        )
        assert r.resolved.program is not None

    def test_enum_payload_variant_inferred_from_annotation(self) -> None:
        r = accept_type(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "let x: Option[int] = some(value = 1)\nx"
        )
        assert r.resolved.program is not None

    def test_nullary_variant_inferred_from_annotation(self) -> None:
        r = accept_type(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "let x: Option[int] = none()\nx"
        )
        assert r.resolved.program is not None

    def test_record_inferred_from_list_element_context(self) -> None:
        r = accept_type(
            "record Box[T]\n"
            "  value: T\n"
            "let bs: list[Box[int]] = [Box(value = 1), Box(value = 2)]\nbs"
        )
        assert r.resolved.program is not None

    def test_qualified_variant_inferred(self) -> None:
        r = accept_type(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "let x: Option[int] = Option.some(value = 1)\nx"
        )
        assert r.resolved.program is not None

    def test_two_type_params_inferred_from_fields(self) -> None:
        r = accept_type(
            "record Pair[A, B]\n"
            "  first: A\n"
            "  second: B\n"
            'Pair(first = 1, second = "hi")'
        )
        assert r.resolved.program is not None

    def test_inferred_type_matches_result(self) -> None:
        # Box(value = 1) infers T=int; assert the binding's type_args is (IntType(),)
        # to lock down "no stale T / no concrete leak".
        r = accept_type(
            "record Box[T]\n"
            "  value: T\n"
            "let b = Box(value = 1)\nb"
        )
        decl = r.resolved.program.body.items[1]
        assert isinstance(decl, LetDecl)
        binding = r.type_env.get_binding_type(decl.node_id)
        assert isinstance(binding, RecordType)
        assert binding.type_args == (IntType(),)

    def test_uninferable_type_var_rejected(self) -> None:
        # no fields, no annotation -> T cannot be solved
        err = reject_type(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "none()"
        )
        assert (
            "infer" in str(err).lower()
            or "type argument" in str(err).lower()
            or "annotation" in str(err).lower()
        )


class TestGenericConstructorExplicit:
    """Tests for explicit type arguments on generic constructors."""

    def test_record_explicit_type_arg(self) -> None:
        r = accept_type(
            "record Box[T]\n"
            "  value: T\n"
            "Box::[int](value = 42)"
        )
        assert r.resolved.program is not None

    def test_enum_variant_explicit_type_arg(self) -> None:
        r = accept_type(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "some::[int](value = 1)"
        )
        assert r.resolved.program is not None

    def test_qualified_enum_variant_explicit_type_arg(self) -> None:
        r = accept_type(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "Option.some::[int](value = 1)"
        )
        assert r.resolved.program is not None

    def test_two_type_params_explicit(self) -> None:
        r = accept_type(
            "record Pair[A, B]\n"
            "  first: A\n"
            "  second: B\n"
            'Pair::[int, text](first = 1, second = "hi")'
        )
        assert r.resolved.program is not None

    def test_explicit_wrong_arity_rejected(self) -> None:
        err = reject_type(
            "record Box[T]\n"
            "  value: T\n"
            "Box::[int, text](value = 42)"
        )
        assert "type argument" in str(err).lower()

    def test_nullary_variant_explicit_type_arg(self) -> None:
        r = accept_type(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "none::[int]()"
        )
        assert r.resolved.program is not None


class TestGenericConstructorErrors:
    """Error cases for generic constructors."""

    def test_field_type_mismatch_rejected(self) -> None:
        err = reject_type(
            "record Box[T]\n"
            "  value: T\n"
            'let b: Box[int] = Box(value = "text")\nb'
        )
        assert "mismatch" in str(err).lower() or "expected" in str(err).lower()

    def test_missing_field_rejected(self) -> None:
        err = reject_type(
            "record Box[T]\n"
            "  value: T\n"
            "Box()"
        )
        assert (
            "missing" in str(err).lower()
            or "field" in str(err).lower()
            or "infer" in str(err).lower()
        )

    def test_unknown_field_rejected(self) -> None:
        err = reject_type(
            "record Box[T]\n"
            "  value: T\n"
            "Box(value = 1, extra = 2)"
        )
        assert (
            "no field" in str(err).lower()
            or "field" in str(err).lower()
            or "unknown" in str(err).lower()
        )

    def test_inconsistent_type_inference_rejected(self) -> None:
        # Pair(first: 1, second: T=bool) when annotation says Pair[int, int]
        err = reject_type(
            "record Pair[A, B]\n"
            "  first: A\n"
            "  second: B\n"
            "let p: Pair[int, int] = Pair(first = 1, second = true)\np"
        )
        assert (
            "mismatch" in str(err).lower()
            or "expected" in str(err).lower()
            or "bool" in str(err).lower()
        )

    def test_positional_arg_rejected(self) -> None:
        err = reject_type(
            "record Box[T]\n"
            "  value: T\n"
            "Box(42)"
        )
        assert "named" in str(err).lower() or "positional" in str(err).lower()


class TestGenericInvariance:
    """Tests for invariant type argument checking."""

    def test_box_int_not_assignable_to_box_text(self) -> None:
        err = reject_type(
            "record Box[T]\n"
            "  value: T\n"
            'let b: Box[text] = Box(value = 1)\nb'
        )
        assert (
            "mismatch" in str(err).lower()
            or "expected" in str(err).lower()
            or "int" in str(err).lower()
        )

    def test_option_int_not_assignable_to_option_text(self) -> None:
        err = reject_type(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "let x: Option[text] = some(value = 1)\nx"
        )
        assert (
            "mismatch" in str(err).lower()
            or "expected" in str(err).lower()
            or "int" in str(err).lower()
        )

    def test_box_int_assignable_to_box_int(self) -> None:
        r = accept_type(
            "record Box[T]\n"
            "  value: T\n"
            "let b: Box[int] = Box(value = 1)\nb"
        )
        assert r.resolved.program is not None


class TestGenericFieldAccess:
    """Tests for field access on generic record instances."""

    def test_field_access_on_generic_record(self) -> None:
        r = accept_type(
            "record Box[T]\n"
            "  value: T\n"
            "let b: Box[int] = Box(value = 42)\nb.value"
        )
        assert r.resolved.program is not None

    def test_field_type_is_instantiated(self) -> None:
        r = accept_type(
            "record Box[T]\n"
            "  value: T\n"
            "let b: Box[int] = Box(value = 42)\n"
            "let v: int = b.value\nv"
        )
        assert r.resolved.program is not None

    def test_field_type_mismatch_after_instantiation(self) -> None:
        err = reject_type(
            "record Box[T]\n"
            "  value: T\n"
            "let b: Box[int] = Box(value = 42)\n"
            "let s: text = b.value\ns"
        )
        assert "mismatch" in str(err).lower() or "expected" in str(err).lower()

    def test_two_param_field_access(self) -> None:
        r = accept_type(
            "record Pair[A, B]\n"
            "  first: A\n"
            "  second: B\n"
            'let p: Pair[int, text] = Pair(first = 1, second = "hi")\n'
            "let x: int = p.first\n"
            "let y: text = p.second\nx"
        )
        assert r.resolved.program is not None


class TestGenericPatterns:
    """Tests for pattern matching on generic enum instances."""

    def test_case_on_generic_enum(self) -> None:
        r = accept_type(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "let x: Option[int] = some(value = 1)\n"
            "case x of | some(value = v) => v | none() => 0"
        )
        assert r.resolved.program is not None

    def test_pattern_field_type_is_instantiated(self) -> None:
        r = accept_type(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "let x: Option[int] = some(value = 1)\n"
            "case x of | some(value = v) => v + 1 | none() => 0"
        )
        assert r.resolved.program is not None


class TestGenericConstructorAsValue:
    """Tests for generic constructors used as values (not in direct call position)."""

    def test_payload_constructor_as_value_with_annotation(self) -> None:
        r = accept_type(
            "record Box[T]\n"
            "  value: T\n"
            "let mk: (int) -> Box[int] = Box\nmk"
        )
        assert r.resolved.program is not None

    def test_nullary_variant_as_value_with_annotation(self) -> None:
        r = accept_type(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "let x: Option[int] = none\nx"
        )
        assert r.resolved.program is not None

    def test_generic_constructor_as_value_no_context_rejected(self) -> None:
        err = reject_type(
            "record Box[T]\n"
            "  value: T\n"
            "Box"
        )
        assert (
            "infer" in str(err).lower()
            or "type argument" in str(err).lower()
            or "annotation" in str(err).lower()
        )


class TestNonGenericConstructorsUnchanged:
    """Verify that non-generic constructors continue to work as before."""

    def test_non_generic_record_still_works(self) -> None:
        r = accept_type("record Point\n  x: int\n  y: int\nPoint(x = 1, y = 2)")
        assert r.resolved.program is not None

    def test_non_generic_enum_still_works(self) -> None:
        r = accept_type("enum Color\n  | Red\n  | Blue\nRed()")
        assert r.resolved.program is not None

    def test_non_generic_type_arg_rejected(self) -> None:
        err = reject_type("record Point\n  x: int\n  y: int\nPoint::[int](x = 1, y = 2)")
        assert (
            "type argument" in str(err).lower()
            or "not a generic" in str(err).lower()
            or "generic" in str(err).lower()
        )

    def test_non_generic_enum_variant_type_arg_rejected(self) -> None:
        err = reject_type("enum Color\n  | Red\nRed::[int]()")
        assert (
            "type argument" in str(err).lower()
            or "not a generic" in str(err).lower()
            or "generic" in str(err).lower()
        )


class TestNonGenericConstructorAsValue:
    """Non-generic constructor with fields used in value position."""

    def test_record_ctor_with_fields_as_value_is_function_type(self) -> None:
        r = accept_type(
            "record Box\n  item: int\nlet make = Box\nmake"
        )
        prog = r.resolved.program
        assert prog is not None
        make_ref = prog.body.items[-1]
        assert isinstance(make_ref, VarRef)
        t = r.node_types[make_ref.node_id]
        assert isinstance(t, FunctionType)
        assert t.params == (IntType(),)
        assert isinstance(t.result, RecordType)
        assert t.result.name == "Box"

    def test_record_ctor_value_called_positionally(self) -> None:
        r = accept_type(
            "record Box\n  item: int\nlet make = Box\nmake(5)"
        )
        assert r.resolved.program is not None

    def test_record_ctor_value_called_with_named_arg_rejected(self) -> None:
        err = reject_type(
            "record Box\n  item: int\nlet make = Box\nmake(item = 5)"
        )
        assert (
            "named" in str(err).lower()
            or "positional" in str(err).lower()
            or "keyword" in str(err).lower()
        )

    def test_record_ctor_value_arity_mismatch_rejected(self) -> None:
        err = reject_type(
            "record Box\n  item: int\nlet make = Box\nmake(1, 2)"
        )
        assert "argument" in str(err).lower() or "too many" in str(err).lower()

    def test_enum_payload_variant_as_value_is_function_type(self) -> None:
        r = accept_type(
            "enum E\n  | Nope\n  | Wrap(value: int)\nlet w = Wrap\nw"
        )
        prog = r.resolved.program
        assert prog is not None
        w_ref = prog.body.items[-1]
        assert isinstance(w_ref, VarRef)
        t = r.node_types[w_ref.node_id]
        assert isinstance(t, FunctionType)
        assert t.params == (IntType(),)
        assert isinstance(t.result, EnumType)
        assert t.result.name == "E"

    def test_qualified_enum_payload_variant_as_value_is_function_type(self) -> None:
        r = accept_type(
            "enum E\n  | Nope\n  | Wrap(value: int)\nlet w = E.Wrap\nw"
        )
        prog = r.resolved.program
        assert prog is not None
        w_ref = prog.body.items[-1]
        assert isinstance(w_ref, VarRef)
        t = r.node_types[w_ref.node_id]
        assert isinstance(t, FunctionType)
        assert t.params == (IntType(),)
        assert isinstance(t.result, EnumType)
        assert t.result.name == "E"

    def test_nullary_variant_as_value_stays_nominal(self) -> None:
        r = accept_type("enum E\n  | Nope\n  | Wrap(value: int)\nlet n = Nope\nn")
        prog = r.resolved.program
        assert prog is not None
        n_ref = prog.body.items[-1]
        assert isinstance(n_ref, VarRef)
        t = r.node_types[n_ref.node_id]
        assert isinstance(t, EnumType)
        assert t.name == "E"

    def test_exception_ctor_as_value_rejected(self) -> None:
        err = reject_type("let f = Abort\nf")
        assert "exception" in str(err).lower()
        assert "first-class" in str(err).lower() or "directly" in str(err).lower()

    def test_qualified_record_field_as_value_rejected(self) -> None:
        # A `RecordName.field` reference in value position routes through the
        # qualified-constructor-as-value path; the record is not an enum.
        err = reject_type('record Box\n  item: int\nlet f = Box.item\n"x"')
        assert "not a known enum" in str(err).lower()

    def test_qualified_unknown_variant_as_value_rejected(self) -> None:
        # An `EnumName.bogus` reference in value position names a variant that
        # does not exist; scope defers this to the checker.
        err = reject_type(
            'enum E\n  | Nope\n  | Wrap(value: int)\nlet f = E.Bogus\n"x"'
        )
        assert "does not exist in enum" in str(err).lower()


class TestGenericEnumQualifiersAndTypeVarScoping:
    """Regression: qualified patterns/`is`-tests on generic enums, and type
    variables of an enclosing generic ``def`` visible in nested annotations."""

    _OPTION = "enum Option[T]\n  | none\n  | some(value: T)\n"

    def test_qualified_pattern_on_generic_enum(self) -> None:
        # Previously: "'Option' is not a known enum type." — the qualifier was
        # resolved as a concrete type, which rejects bare generic names.
        r = accept_type(
            self._OPTION
            + "let o: Option[int] = some(value = 5)\n"
            + "case o of\n"
            + "  | Option.none => 0\n"
            + "  | Option.some(value) => value\n"
        )
        assert r.resolved.program is not None

    def test_qualified_pattern_generic_enum_binds_instantiated_field(self) -> None:
        # The bound payload is instantiated from the scrutinee's type args.
        r = accept_type(
            self._OPTION
            + "let o: Option[int] = some(value = 5)\n"
            + "case o of\n"
            + "  | Option.none => 0\n"
            + "  | Option.some(value) => value + 1\n"
        )
        assert r.resolved.program is not None

    def test_qualified_is_test_on_generic_enum(self) -> None:
        r = accept_type(
            self._OPTION
            + "let o: Option[int] = some(value = 5)\n"
            + "if o is Option.some => print 1\n"
            + "if o is not Option.none => print 2\n"
        )
        assert r.resolved.program is not None

    def test_generic_record_name_as_variant_qualifier_rejected(self) -> None:
        # A generic *record* used as a variant qualifier on an enum scrutinee
        # is not a known enum type (covers the non-enum generic-kind branch).
        err = reject_type(
            self._OPTION
            + "record Box[T]\n  value: T\n"
            + "let o: Option[int] = some(value = 5)\n"
            + "if o is Box.some => print 1\n"
        )
        assert "not a known enum type" in str(err).lower()

    def test_mismatched_generic_enum_qualifier_rejected(self) -> None:
        # A qualifier naming a different generic enum than the scrutinee.
        err = reject_type(
            self._OPTION
            + "enum Maybe[T]\n  | nothing\n  | just(value: T)\n"
            + "let o: Option[int] = some(value = 5)\n"
            + "if o is Maybe.just => print 1\n"
        )
        assert "resolves to enum" in str(err).lower()

    def test_type_var_lambda_annotation_in_generic_def_body(self) -> None:
        # A lambda inside a generic def may annotate its params/return with the
        # def's rigid type variables. Previously: "Unknown type 'A'".
        r = accept_type(
            "def idf[A](x: A) -> A =\n"
            "  let g: (A) -> A = fn(y: A) -> A => y\n"
            "  g(x)\n"
            "let r = idf(7)\nr"
        )
        assert r.resolved.program is not None

    def test_type_var_let_annotation_in_generic_def_body(self) -> None:
        # A `let` annotation in a generic def body may reference a type variable.
        r = accept_type(
            "def mk[A](x: A) -> list[A] =\n"
            "  let single: list[A] = [x]\n"
            "  single\n"
            "let r = mk(3)\nr"
        )
        assert r.resolved.program is not None

    def test_type_var_lambda_return_type_in_generic_def(self) -> None:
        r = accept_type(
            "def pickfn[A]() -> (A) -> A = fn(y: A) -> A => y\n"
            '"ok"'
        )
        assert r.resolved.program is not None


# ---------------------------------------------------------------------------
# additional coverage tests
# ---------------------------------------------------------------------------


class TestGenericCoverageEdgeCases:
    """Tests for code paths not yet exercised by the main test classes."""

    def test_generic_field_type_references_generic_record(self) -> None:
        """AppliedT branch (record path) in _ensure_referenced_type_built."""
        r = accept_type(
            "record Box[T]\n"
            "  value: T\n"
            "record Wrapper\n"
            "  box: Box[int]\n"
            "let w = Wrapper(box = Box(value = 42))\nw"
        )
        assert r.resolved.program is not None

    def test_generic_field_type_references_generic_enum(self) -> None:
        """AppliedT branch (enum path) in _ensure_referenced_type_built."""
        r = accept_type(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "record Wrapper\n"
            "  opt: Option[int]\n"
            "let w = Wrapper(opt = some(value = 1))\nw"
        )
        assert r.resolved.program is not None

    def test_duplicate_field_in_generic_enum_variant_rejected(self) -> None:
        """Duplicate field check in generic enum variant."""
        err = reject_type("enum E[T]\n  | A(x: T, x: T)")
        assert "duplicate" in str(err).lower() or "field" in str(err).lower()


    def test_generic_constructor_as_value_no_context_rejected(self) -> None:
        """Unsolvable type param error in _check_generic_constructor_as_value."""
        err = reject_type(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "none"
        )
        assert (
            "infer" in str(err).lower()
            or "annotation" in str(err).lower()
            or "type argument" in str(err).lower()
        )

    def test_applied_t_with_unknown_name_in_field(self) -> None:
        """AppliedT where name is not a record/enum triggers a resolve error."""
        # Box[T] is a generic record; Wrapper has a field of type Box[Unknown[int]].
        # When building Wrapper, _ensure_referenced_type_built is called on the field type.
        # Box is found in record_defs but Unknown is not in record_defs or
        # enum_defs, so the error surfaces in resolve_type_expr.
        err = reject_type(
            "record Box[T]\n"
            "  value: T\n"
            "record Wrapper\n"
            "  box: Box[Unknown[int]]\n"
            "let w = Wrapper(box = Box(value = 1))\nw"
        )
        assert err is not None


# ---------------------------------------------------------------------------
# Fix 1: nested-generic inference (regression tests)
# ---------------------------------------------------------------------------


class TestNestedGenericInference:
    """Regression tests for _match recursing into generic RecordType/EnumType type_args.

    These verify that nested-generic inference works WITHOUT explicit ::[…] or annotations.
    """

    def test_def_call_unwrap_infers_u_from_box(self) -> None:
        # def-call path: unwrap(b = Box(value = 1)) must infer U=int without annotation.
        r = accept_type(
            "record Box[T]\n"
            "  value: T\n"
            "def unwrap[U](b: Box[U]) -> U = b.value\n"
            "let n: int = unwrap(b = Box(value = 1))\nn"
        )
        assert r.resolved.program is not None

    def test_def_call_unwrap_wrong_return_type_rejected(self) -> None:
        # unwrap returns U=int; annotating as text must be rejected.
        err = reject_type(
            "record Box[T]\n"
            "  value: T\n"
            "def unwrap[U](b: Box[U]) -> U = b.value\n"
            "let s: text = unwrap(b = Box(value = 1))\ns"
        )
        assert "mismatch" in str(err).lower() or "expected" in str(err).lower()

    def test_constructor_holder_infers_u_from_nested_box(self) -> None:
        # constructor path: Holder(inner = Box(value = 1)) infers U=int.
        # Assert the binding's type_args is (IntType(),).
        r = accept_type(
            "record Box[T]\n"
            "  value: T\n"
            "record Holder[U]\n"
            "  inner: Box[U]\n"
            "let h = Holder(inner = Box(value = 1))\nh"
        )
        decl = r.resolved.program.body.items[2]
        assert isinstance(decl, LetDecl)
        binding = r.type_env.get_binding_type(decl.node_id)
        assert isinstance(binding, RecordType)
        assert binding.type_args == (IntType(),)

    def test_constructor_holder_use_as_int_works(self) -> None:
        # Inferred Holder[int] — accessing inner.value as int must succeed.
        r = accept_type(
            "record Box[T]\n"
            "  value: T\n"
            "record Holder[U]\n"
            "  inner: Box[U]\n"
            "let h = Holder(inner = Box(value = 1))\n"
            "let n: int = h.inner.value\nn"
        )
        assert r.resolved.program is not None

    def test_inconsistent_nested_inference_rejected(self) -> None:
        # If the same type var U is bound to both int and text, must be rejected.
        err = reject_type(
            "record Box[T]\n"
            "  value: T\n"
            "def swap[U](a: Box[U], b: Box[U]) -> U = a.value\n"
            "swap(a = Box(value = 1), b = Box(value = \"hi\"))"
        )
        assert (
            "inconsistent" in str(err).lower()
            or "mismatch" in str(err).lower()
            or "expected" in str(err).lower()
        )


# ---------------------------------------------------------------------------
# Fix 2: generic-recursion rejection tests
# ---------------------------------------------------------------------------


class TestGenericRecursionRejection:
    """Tests asserting the 'directly or indirectly recursive' error for generic types."""

    def test_generic_record_direct_recursion_rejected(self) -> None:
        # record Tree[T] with a field of type Tree[T] is directly recursive.
        err = reject_type(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "record Tree[T]\n"
            "  value: T\n"
            "  child: Tree[T]\n"
            "Tree(value = 1, child = Tree(value = 2, child = ()))"
        )
        assert "recursive" in str(err).lower()

    def test_generic_record_indirect_via_list_rejected(self) -> None:
        # children: list[Tree[T]] is indirect recursion via list.
        err = reject_type(
            "record Tree[T]\n"
            "  value: T\n"
            "  children: list[Tree[T]]\n"
            "Tree(value = 1, children = [])"
        )
        assert "recursive" in str(err).lower()

    def test_generic_record_indirect_via_dict_rejected(self) -> None:
        # dict[text, Tree[T]] is indirect recursion via dict.
        err = reject_type(
            "record Tree[T]\n"
            "  value: T\n"
            "  children: dict[text, Tree[T]]\n"
            "Tree(value = 1, children = {})"
        )
        assert "recursive" in str(err).lower()

    def test_generic_enum_direct_recursion_rejected(self) -> None:
        # enum L[T] | nil | cons(tail: L[T]) is directly recursive.
        err = reject_type(
            "enum L[T]\n"
            "  | nil\n"
            "  | cons(head: T, tail: L[T])\n"
            "nil()"
        )
        assert "recursive" in str(err).lower()

    def test_generic_mutual_recursion_rejected(self) -> None:
        # record A[T] with field b: B[T] and record B[T] with field a: A[T].
        err = reject_type(
            "record A[T]\n"
            "  b: B[T]\n"
            "record B[T]\n"
            "  a: A[T]\n"
            "A(b = B(a = ()))"
        )
        assert "recursive" in str(err).lower()


# ---------------------------------------------------------------------------
# Fix 3: abstract-instance field access and pattern tests in generic defs
# ---------------------------------------------------------------------------


class TestGenericAbstractInstanceAccess:
    """Tests for field access and pattern matching on abstract generic instances inside defs."""

    def test_field_access_in_generic_def_yields_type_var(self) -> None:
        # def unbox[U](b: Box[U]) -> U = b.value; field access on Box[U] yields U.
        r = accept_type(
            "record Box[T]\n"
            "  value: T\n"
            "def unbox[U](b: Box[U]) -> U = b.value\n"
            "let n: int = unbox(b = Box(value = 42))\nn"
        )
        assert r.resolved.program is not None

    def test_field_access_wrong_return_type_rejected(self) -> None:
        # unbox returns U; annotating as text when U=int must be rejected.
        err = reject_type(
            "record Box[T]\n"
            "  value: T\n"
            "def unbox[U](b: Box[U]) -> U = b.value\n"
            "let s: text = unbox(b = Box(value = 42))\ns"
        )
        assert "mismatch" in str(err).lower() or "expected" in str(err).lower()

    def test_case_on_generic_enum_in_def_binds_type_var(self) -> None:
        # A generic def that extracts a value from Option[U] via case.
        r = accept_type(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "def get_or[U](opt: Option[U], default: U) -> U =\n"
            "  case opt of\n"
            "    | some(value = v) => v\n"
            "    | none() => default\n"
            "let n: int = get_or(opt = some(value = 1), default = 0)\nn"
        )
        assert r.resolved.program is not None

    def test_case_on_generic_enum_wrong_return_type_rejected(self) -> None:
        # The generic def returns U; mismatching annotation must be rejected.
        err = reject_type(
            "enum Option[T]\n"
            "  | none\n"
            "  | some(value: T)\n"
            "def get_or[U](opt: Option[U], default: U) -> U =\n"
            "  case opt of\n"
            "    | some(value = v) => v\n"
            "    | none() => default\n"
            "let s: text = get_or(opt = some(value = 1), default = 0)\ns"
        )
        assert "mismatch" in str(err).lower() or "expected" in str(err).lower()


# ---------------------------------------------------------------------------
# Cast typecheck tests
# ---------------------------------------------------------------------------


class TestCast:
    """Tests for the Cast node type checking."""

    def test_int_as_text_total_render(self) -> None:
        """int as text yields text."""
        r = accept_type("let s: text = 1 as text\ns")
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        assert r.node_types[decl.value.node_id] == TextType()

    def test_text_as_json_total_json(self) -> None:
        """text as json yields json."""
        r = accept_type('let j: json = "hello" as json\nj')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        assert r.node_types[decl.value.node_id] == JsonType()

    def test_text_as_int_fallible_yields_int(self) -> None:
        """text as int yields int type."""
        r = accept_type('let x: int = "42" as int\nx')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        assert r.node_types[decl.value.node_id] == IntType()

    def test_decimal_as_int_fallible(self) -> None:
        """decimal as int is fallible, yields int."""
        r = accept_type("let d = 3.5\nlet x: int = d as int\nx")
        decl = r.resolved.program.body.items[1]
        assert isinstance(decl, LetDecl)
        assert isinstance(decl.value, Cast)
        assert r.node_types[decl.value.node_id] == IntType()

    def test_int_as_decimal_noop(self) -> None:
        """int as decimal is a no-op widen."""
        r = accept_type("let d: decimal = 1 as decimal\nd")
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        assert r.node_types[decl.value.node_id] == DecimalType()

    def test_as_question_yields_bool(self) -> None:
        """as? always yields bool."""
        r = accept_type('let b: bool = "42" as? int\nb')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        assert r.node_types[decl.value.node_id] == BoolType()

    def test_as_question_on_total_cast_yields_bool(self) -> None:
        """as? on a total cast also yields bool."""
        r = accept_type("let b: bool = 1 as? text\nb")
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        assert r.node_types[decl.value.node_id] == BoolType()

    def test_bool_to_int_rejected(self) -> None:
        """bool as int is a static error."""
        err = reject_type("true as int")
        assert "bool" in str(err).lower() or "cannot cast" in str(err).lower()

    def test_int_to_bool_rejected(self) -> None:
        """int as bool is a static error."""
        err = reject_type("1 as bool")
        assert "cannot cast" in str(err).lower() or "int" in str(err).lower()

    def test_agent_as_text_rejected(self) -> None:
        """agent value as text is a static error."""
        err = reject_type("agent myAgent\nmyAgent as text")
        assert "cannot cast" in str(err).lower() or "agent" in str(err).lower()

    def test_record_as_json_accepted(self) -> None:
        """record as json is now TOTAL_JSON (explicit nominal→json cast)."""
        from agm.agl.semantics.types import CastKind
        r = accept_type("record R\n  x: int\nlet r = R(x = 1)\nr as json")
        assert r
        spec = next(iter(r.cast_specs.values()))
        assert spec.kind == CastKind.TOTAL_JSON

    def test_cast_to_unit_rejected(self) -> None:
        """casting to unit is a static error."""
        err = reject_type("1 as unit")
        assert "cannot cast" in str(err).lower() or "unit" in str(err).lower()

    def test_as_question_same_static_error(self) -> None:
        """as? on a static-error pair is also a static error."""
        err = reject_type("true as? int")
        assert "cannot cast" in str(err).lower()

    def test_json_to_text_render(self) -> None:
        """json as text yields text (TOTAL_RENDER —  completeness)."""
        r = accept_type('let j: json = 42\nlet s: text = j as text\ns')
        decl = r.resolved.program.body.items[1]
        assert isinstance(decl, LetDecl)
        assert isinstance(decl.value, Cast)
        assert r.node_types[decl.value.node_id] == TextType()

    def test_json_as_text_let_binding(self) -> None:
        """json as text typechecks: let s: text = (1 as json) as text."""
        r = accept_type("let s: text = (1 as json) as text\ns")
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        assert r.node_types[decl.value.node_id] == TextType()

    def test_chained_cast_int_to_json_to_text(self) -> None:
        """Chained x as json as text resolves to text ( example)."""
        r = accept_type("let x: int = 1\nlet s: text = x as json as text\ns")
        decl = r.resolved.program.body.items[1]
        assert isinstance(decl, LetDecl)
        assert isinstance(decl.value, Cast)
        assert r.node_types[decl.value.node_id] == TextType()

    def test_json_to_list_fallible(self) -> None:
        """json as list[int] yields list[int]."""
        r = accept_type("let j: json = 42\nlet xs: list[int] = j as list[int]\nxs")
        decl = r.resolved.program.body.items[1]
        assert isinstance(decl, LetDecl)
        assert isinstance(decl.value, Cast)
        assert r.node_types[decl.value.node_id] == ListType(elem=IntType())

    def test_cast_spec_stored(self) -> None:
        """CastSpec is stored in CheckedProgram.cast_specs."""
        from agm.agl.semantics.types import CastKind
        r = accept_type("1 as text")
        assert len(r.cast_specs) == 1
        spec = next(iter(r.cast_specs.values()))
        assert spec.kind == CastKind.TOTAL_RENDER

    def test_as_question_spec_stored(self) -> None:
        """as? CastSpec is stored with same kind as as."""
        from agm.agl.semantics.types import CastKind
        r = accept_type('"hello" as? int')
        assert len(r.cast_specs) == 1
        spec = next(iter(r.cast_specs.values()))
        assert spec.kind == CastKind.FALLIBLE

    def test_raise_operand_can_be_cast(self) -> None:
        """A cast preserves the universal assignability of bottom."""
        r = accept_type('(raise Abort(message = "x")) as text')
        cast = r.resolved.program.body.items[0]
        assert isinstance(cast, Cast)
        assert r.node_types[cast.node_id] == TextType()


class TestCastClassificationTable:
    """Direct unit tests for cast_classification() covering the new nominal→json pairs."""

    def test_record_to_json_is_total_json(self) -> None:
        from agm.agl.semantics.types import CastKind, cast_classification

        source = RecordType(name="R", fields={"x": IntType()})
        assert cast_classification(source, JsonType()) == CastKind.TOTAL_JSON

    def test_enum_to_json_is_total_json(self) -> None:
        from agm.agl.semantics.types import CastKind, cast_classification

        source = EnumType(name="E", variants={"A": {}, "B": {}})
        assert cast_classification(source, JsonType()) == CastKind.TOTAL_JSON

    def test_exception_to_json_is_total_json(self) -> None:
        from agm.agl.semantics.types import CastKind, cast_classification

        source = ExceptionType(name="Abort", fields={"message": TextType(), "trace_id": TextType()})
        assert cast_classification(source, JsonType()) == CastKind.TOTAL_JSON

    def test_exception_to_text_is_total_render(self) -> None:
        from agm.agl.semantics.types import CastKind, cast_classification

        source = ExceptionType(name="Abort", fields={"message": TextType(), "trace_id": TextType()})
        assert cast_classification(source, TextType()) == CastKind.TOTAL_RENDER


class TestParseJsonCall:
    """Tests for parse_json built-in."""

    def test_parse_json_returns_json(self) -> None:
        """parse_json("...") yields json."""
        r = accept_type('let j: json = parse_json("42")\nj')
        assert r

    def test_parse_json_named_arg_rejected(self) -> None:
        """Named args to parse_json are rejected."""
        err = reject_type('parse_json(text = "42")')
        assert "parse_json" in str(err).lower() or "positional" in str(err).lower()

    def test_parse_json_wrong_arity_rejected(self) -> None:
        """parse_json() with wrong arity is rejected."""
        err = reject_type('parse_json("a", "b")')
        assert "parse_json" in str(err).lower()

    def test_parse_json_no_args_rejected(self) -> None:
        """parse_json() with no args is rejected."""
        err = reject_type("parse_json()")
        assert "parse_json" in str(err).lower()


class TestImportDeclTypecheck:
    """Import declarations pass through the type-checker without errors."""

    def test_import_decl_does_not_raise(self) -> None:
        """A bare import declaration type-checks as unit."""
        r = accept_type("import foo.bar\n1")
        assert r  # no exception

    def test_import_with_alias_does_not_raise(self) -> None:
        r = accept_type("import foo as f\n1")
        assert r

    def test_import_wildcard_does_not_raise(self) -> None:
        r = accept_type("import foo.*\n1")
        assert r

    def test_import_using_does_not_raise(self) -> None:
        r = accept_type("import foo using bar\n1")
        assert r

    def test_import_hiding_does_not_raise(self) -> None:
        r = accept_type("import foo hiding secret\n1")
        assert r


class TestSelfRefTypeInSingleModule:
    """Self-reference type annotations (::Name) in single-module programs."""

    def test_self_ref_own_record_type_accepted(self) -> None:
        """'::MyRecord' in a single-module program resolves to the local record type."""
        r = accept_type("record MyRecord\n  x: int\ndef f() -> ::MyRecord = MyRecord(x = 1)\nf()")
        # accept_type returns CheckedProgram; just checking it doesn't raise
        assert r is not None

    def test_self_ref_unknown_type_rejected(self) -> None:
        """'::NoSuch' when NoSuch is not declared → type error."""
        reject_type("def f() -> ::NoSuch = 1\nf()")

    def test_module_qualifier_in_single_module_rejected(self) -> None:
        """'mylib::Point' in single-module (no graph) mode → type error."""
        reject_type("def f() -> mylib::Point = 1\nf()")


# ---------------------------------------------------------------------------
# BUG 1 regression: module_id on generic nominal instances
# instantiate_from_gdef must stamp the result with the template's module_id,
# not with the default ENTRY_ID.
# ---------------------------------------------------------------------------


class TestGenericNominalModuleId:
    """Regression tests for BUG 1: module_id dropped when instantiating generic nominals."""

    def test_instantiate_from_gdef_preserves_record_module_id(self) -> None:
        """instantiate_from_gdef stamps RecordType result with gdef.template.module_id.

        Before the fix, the result always got module_id=ENTRY_ID regardless of
        which module the template belonged to.  After the fix, the result's
        module_id matches the template's module_id.
        """
        from agm.agl.modules.ids import ENTRY_ID, ModuleId
        from agm.agl.semantics.types import TypeVarType
        from agm.agl.typecheck.env import GenericTypeDef

        lib_id = ModuleId.from_dotted("mylib")
        template = RecordType("Box", {"value": TypeVarType("T")}, module_id=lib_id)
        gdef = GenericTypeDef(kind="record", type_params=("T",), template=template)
        env = TypeEnvironment()
        result = env.instantiate_from_gdef("Box", gdef, (IntType(),))
        assert isinstance(result, RecordType)
        assert result.module_id == lib_id, (
            f"Expected module_id={lib_id!r}, got {result.module_id!r}. "
            "instantiate_from_gdef must preserve the template's module_id."
        )
        assert result.module_id != ENTRY_ID

    def test_instantiate_from_gdef_preserves_enum_module_id(self) -> None:
        """instantiate_from_gdef stamps EnumType result with gdef.template.module_id."""
        from agm.agl.modules.ids import ENTRY_ID, ModuleId
        from agm.agl.semantics.types import TypeVarType
        from agm.agl.typecheck.env import GenericTypeDef

        lib_id = ModuleId.from_dotted("mylib")
        template = EnumType(
            "Option",
            {"Some": {"value": TypeVarType("T")}, "None": {}},
            module_id=lib_id,
        )
        gdef = GenericTypeDef(kind="enum", type_params=("T",), template=template)
        env = TypeEnvironment()
        result = env.instantiate_from_gdef("Option", gdef, (IntType(),))
        assert isinstance(result, EnumType)
        assert result.module_id == lib_id, (
            f"Expected module_id={lib_id!r}, got {result.module_id!r}. "
            "instantiate_from_gdef must preserve the template's module_id."
        )
        assert result.module_id != ENTRY_ID

    def test_same_name_generic_different_module_id_not_equal(self) -> None:
        """Box[int] from two modules with different module_ids must NOT be equal.

        This directly tests the invariant that nominal identity includes module_id.
        """
        from agm.agl.modules.ids import ModuleId
        from agm.agl.semantics.types import TypeVarType
        from agm.agl.typecheck.env import GenericTypeDef

        lib_a = ModuleId.from_dotted("libA")
        lib_b = ModuleId.from_dotted("libB")
        tmpl_a = RecordType("Box", {"value": TypeVarType("T")}, module_id=lib_a)
        tmpl_b = RecordType("Box", {"value": TypeVarType("T")}, module_id=lib_b)
        gdef_a = GenericTypeDef(kind="record", type_params=("T",), template=tmpl_a)
        gdef_b = GenericTypeDef(kind="record", type_params=("T",), template=tmpl_b)
        env = TypeEnvironment()
        inst_a = env.instantiate_from_gdef("Box", gdef_a, (IntType(),))
        inst_b = env.instantiate_from_gdef("Box", gdef_b, (IntType(),))
        assert inst_a != inst_b, (
            "Box[int] from libA and Box[int] from libB must be distinct types. "
            "Both had module_id=ENTRY_ID before the fix."
        )

    def test_build_generic_record_stamps_module_id(self) -> None:
        """_TypeBuilder._build_generic_record stamps the template with module_id.

        The checker's _build_generic_record must pass module_id=self._module_id
        when constructing the template RecordType.  Verify via parse+check.
        """
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.typecheck.env import GenericTypeDef

        # In single-module mode the module_id is ENTRY_ID; verify the template
        # also has ENTRY_ID (consistency check: it's at least not wrong).
        cp = parse_resolve_check("record Box[T]\n  value: T\nlet x: Box[int] = Box(value = 1)\nx")
        gdef = cp.type_env.get_generic_type("Box")
        assert gdef is not None
        assert isinstance(gdef, GenericTypeDef)
        assert gdef.template.module_id == ENTRY_ID, (
            f"Template module_id must be ENTRY_ID in single-module mode, "
            f"got {gdef.template.module_id!r}."
        )


# ---------------------------------------------------------------------------
# Field-assignment syntax change: constructor shorthand binding, positional
# literal rejection, and '==' / '!=' equality typing. These specify the TARGET
# behavior and are expected to fail until grammar + constructor checker update.
# ---------------------------------------------------------------------------


class TestFieldAssignmentSyntaxChecks:
    def test_constructor_shorthand_binds_field(self) -> None:
        """R(x) binds field 'x' to the in-scope local 'x' (shorthand for x = x)."""
        r = accept_type("record R\n  x: int\n  y: int\nlet x = 5\nlet r = R(x, y = 1)\nr")
        assert r.resolved.program is not None

    def test_enum_shorthand_binds_field(self) -> None:
        r = accept_type("enum E\n  | A(x: int)\nlet x = 7\nlet e = A(x)\ne")
        assert r.resolved.program is not None

    def test_shorthand_and_explicit_same_field_is_duplicate(self) -> None:
        """A field supplied both via shorthand and explicitly is a duplicate."""
        err = reject_any("record R\n  x: int\nlet x = 1\nR(x, x = 2)")
        assert "duplicate" in str(err).lower()

    def test_reversed_order_shorthand_after_named_arg(self) -> None:
        """Positional args (including bare-name shorthands) must precede named args.
        B(r = ..., x) is rejected because positional 'x' follows named arg 'r = ...'."""
        from agm.agl.parser.errors import AglSyntaxError

        with pytest.raises(AglSyntaxError):
            accept_type(
                "record R\n  x: int\n"
                "enum E\n  | B(x: int, r: R)\n"
                "let x = 1\n"
                "let e = B(r = R(x = 9), x)\ne"
            )

    def test_positional_literal_to_constructor_rejected(self) -> None:
        """A non-VarRef positional arg (a literal) is not shorthand — rejected."""
        err = reject_type("record R\n  x: int\nR(1)")
        assert "named" in str(err).lower() or "positional" in str(err).lower()

    def test_positional_expr_to_constructor_rejected(self) -> None:
        err = reject_type("record R\n  x: int\nlet x = 1\nR(x + 1)")
        assert "named" in str(err).lower() or "positional" in str(err).lower()

    def test_eq_eq_typechecks_to_bool(self) -> None:
        r = accept_type("1 == 2")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == BoolType()

    def test_eq_eq_and_neq_on_structured_value_typecheck_to_bool(self) -> None:
        """Both '==' and '!=' on a structured (record) value yield bool."""
        src = (
            "record P\n  n: int\n"
            "let a = P(n = 1)\n"
            "let b = P(n = 2)\n"
            "a == b\n"
            "a != b"
        )
        r = accept_type(src)
        # item 0 is the RecordDef; the two let bindings are items 1-2; the
        # '==' and '!=' expression statements are items 3 and 4.
        items = r.resolved.program.body.items
        assert r.node_types[items[3].node_id] == BoolType()
        assert r.node_types[items[4].node_id] == BoolType()


# ---------------------------------------------------------------------------
# Finding 1 regression: generic inference for named-only shorthand out-of-order
# ---------------------------------------------------------------------------


class TestGenericNamedOnlyShorthandInference:
    """Regression tests for Finding 1: generic inference must use bind_arguments
    so that named-only shorthand positional args are matched to their param by
    NAME rather than by raw positional index.

    def g[T, U](x: int, *, z: T, w: U) -> T = z

    All three call forms must be accepted and infer correctly.
    """

    _DEF = "def g[T, U](x: int, *, z: T, w: U) -> T = z\n"

    def test_named_only_shorthands_in_order_accepted(self) -> None:
        """g(1, z, w) — in-order shorthands (already worked before fix)."""
        src = self._DEF + "let z: text = \"hi\"\nlet w = 0\ng(1, z, w)"
        r = accept_type(src)
        assert r.resolved.program is not None

    def test_named_only_shorthand_and_named_arg_out_of_order_accepted(self) -> None:
        """g(1, w, z = myv) — shorthand w + named z, out of declaration order."""
        src = self._DEF + "let myv: text = \"hi\"\nlet w = 0\ng(1, w, z = myv)"
        r = accept_type(src)
        assert r.resolved.program is not None

    def test_named_only_two_shorthands_out_of_order_accepted(self) -> None:
        """g(1, w, z) — two shorthands, out of declaration order."""
        src = self._DEF + "let z: text = \"hi\"\nlet w = 0\ng(1, w, z)"
        r = accept_type(src)
        assert r.resolved.program is not None


# ---------------------------------------------------------------------------
# Finding 2 regression: lambda  required-after-defaulted check
# ---------------------------------------------------------------------------


class TestLambdaRequiredAfterDefaulted:
    """Regression tests for Finding 2: the  required-after-defaulted check
    must apply to lambdas, not just def declarations.
    """

    def test_lambda_required_after_defaulted_rejected(self) -> None:
        """fn(x: int = 5, y: int) -> int => x + y must be rejected."""
        err = reject_type("fn(x: int = 5, y: int) -> int => x + y")
        assert "default" in str(err).lower() or "required" in str(err).lower()

    def test_lambda_valid_defaulted_after_required_accepted(self) -> None:
        """fn(x: int, y: int = 5) -> int => x + y is fine (required before default)."""
        r = accept_type("fn(x: int, y: int = 5) -> int => x + y")
        assert r.resolved.program is not None

    def test_lambda_named_only_any_default_order_accepted(self) -> None:
        """Named-only params are order-free: fn(*, x: int = 0, y: int) -> int => y is ok."""
        r = accept_type("fn(*, x: int = 0, y: int) -> int => y")
        assert r.resolved.program is not None


# ---------------------------------------------------------------------------
# Config declarations — typecheck
# ---------------------------------------------------------------------------


class TestConfigDecl:
    def test_config_log_true_ok(self) -> None:
        r = accept_type("config log = true\nprint 1")
        assert r.resolved.program is not None

    def test_config_log_wrong_type_errors(self) -> None:
        reject_any('config log = "yes"\nprint 1')

    def test_config_max_iters_ok(self) -> None:
        r = accept_type("config max-iters = 5\nprint 1")
        assert r.resolved.program is not None

    def test_config_max_iters_wrong_type_errors(self) -> None:
        reject_any('config max-iters = "x"\nprint 1')

    def test_config_strict_json_ok(self) -> None:
        r = accept_type("config strict-json = false\nprint 1")
        assert r.resolved.program is not None

    def test_config_runner_ok(self) -> None:
        r = accept_type('config runner = "claude"\nprint 1')
        assert r.resolved.program is not None

    def test_config_binding_type_recorded(self) -> None:
        from agm.agl.syntax.nodes import ConfigDecl
        r = accept_type("config log = true\nprint 1")
        prog = r.resolved.program
        assert prog is not None
        config_decl = prog.body.items[0]
        assert isinstance(config_decl, ConfigDecl)
        binding_type = r.type_env.get_binding_type(config_decl.node_id)
        assert binding_type == BoolType()
