"""Tests for the AgL v2 type-checking pass (Component 5).

All tests drive real AgL source through ``parse_program`` + ``resolve`` +
``check``, asserting on user-visible behavior: raised ``AglTypeError``
diagnostics and type-table / contract-spec observables via the public
``CheckedProgram`` API.

Grammar notes (shape constraints from the v2 grammar):
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
from agm.agl.syntax.nodes import (
    AgentDecl,
    AssignStmt,
    AssignTarget,
    Block,
    Call,
    Case,
    Constructor,
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
    Program,
    Raise,
    StringLit,
    Try,
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
from agm.agl.typecheck.types import (
    BUILTIN_EXCEPTION_NAMES,
    BUILTIN_PRELUDE_TYPE_NAMES,
    BUILTIN_PRELUDE_TYPES,
    EXCEPTION_BASE,
    comparable_types,
    is_assignable,
    is_json_shaped,
)
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
        assert not comparable_types(BottomType(), IntType())

    def test_not_comparable_right(self) -> None:
        assert not comparable_types(IntType(), BottomType())

    def test_frozen(self) -> None:
        b = BottomType()
        with pytest.raises((AttributeError, TypeError)):
            b.x = 1  # type: ignore[attr-defined]

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


class TestComparableTypes:
    def test_same_int(self) -> None:
        assert comparable_types(IntType(), IntType())

    def test_same_text(self) -> None:
        assert comparable_types(TextType(), TextType())

    def test_same_bool(self) -> None:
        assert comparable_types(BoolType(), BoolType())

    def test_int_decimal_cross(self) -> None:
        assert comparable_types(IntType(), DecimalType())
        assert comparable_types(DecimalType(), IntType())

    def test_agent_not_comparable(self) -> None:
        assert not comparable_types(AgentType(), AgentType())

    def test_function_not_comparable(self) -> None:
        ft = FunctionType(params=(), result=IntType())
        assert not comparable_types(ft, ft)

    def test_unit_not_comparable(self) -> None:
        assert not comparable_types(UnitType(), UnitType())

    def test_bottom_not_comparable_left(self) -> None:
        assert not comparable_types(BottomType(), IntType())

    def test_bottom_not_comparable_right(self) -> None:
        assert not comparable_types(IntType(), BottomType())

    def test_cross_type_not_comparable(self) -> None:
        assert not comparable_types(TextType(), IntType())
        assert not comparable_types(JsonType(), TextType())


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

    def test_print_function_rejected(self) -> None:
        err = reject_type("let f = fn(x: int) -> int => x\nprint(f)")
        assert "function" in str(err).lower() or "rendering" in str(err).lower()

    def test_print_agent_rejected(self) -> None:
        err = reject_type("agent a\nprint(a)")
        assert "agent" in str(err).lower() or "rendering" in str(err).lower()

    def test_print_wrong_arg_count(self) -> None:
        err = reject_type("print(1, 2)")
        assert "print" in str(err).lower() or "argument" in str(err).lower()

    def test_print_no_args(self) -> None:
        err = reject_type("print()")
        assert "print" in str(err).lower() or "argument" in str(err).lower()

    def test_print_named_arg(self) -> None:
        err = reject_type("print(x: 42)")
        assert "print" in str(err).lower() or "argument" in str(err).lower()


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
        r = accept_type('agent reviewer\nask("Q", agent: reviewer)')
        assert r.resolved.program is not None

    def test_ask_no_default_agent_raises(self) -> None:
        err = reject_type('ask("Q")', capabilities=no_agent_caps())
        assert "agent" in str(err).lower() or "default" in str(err).lower()

    def test_ask_no_prompt_raises(self) -> None:
        err = reject_type("ask()")
        assert "prompt" in str(err).lower() or "argument" in str(err).lower()

    def test_ask_wrong_agent_type(self) -> None:
        err = reject_type('let x = "not_agent"\nask("Q", agent: x)')
        assert "agent" in str(err).lower()

    def test_ask_with_json_codec(self) -> None:
        r = accept_type('let n: int = ask("Q", format: "json")\nn')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        spec = r.contract_specs[decl.value.node_id]
        assert spec.codec_name == "json"

    def test_ask_strict_json_true(self) -> None:
        r = accept_type('let n: int = ask("Q", format: "json", strict_json: true)\nn')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        spec = r.contract_specs[decl.value.node_id]
        assert spec.strict_json is True

    def test_ask_strict_json_without_json_codec_raises(self) -> None:
        err = reject_type('let x = ask("Q", strict_json: true)\nx')
        assert "strict_json" in str(err).lower() or "json" in str(err).lower()

    def test_ask_format_non_string_raises(self) -> None:
        err = reject_type('let n: int = ask("Q", format: 42)\nn')
        assert "format" in str(err).lower() or "static" in str(err).lower()

    def test_ask_strict_json_non_bool_raises(self) -> None:
        err = reject_type('let n: int = ask("Q", format: "json", strict_json: "yes")\nn')
        assert "strict_json" in str(err).lower() or "bool" in str(err).lower()

    def test_ask_on_parse_error_abort(self) -> None:
        r = accept_type('let n: int = ask("Q", on_parse_error: Abort())\nn')
        assert r.call_sites[0].parse_policy == "abort"

    def test_ask_on_parse_error_retry(self) -> None:
        r = accept_type('let n: int = ask("Q", on_parse_error: Retry(n: 3))\nn')
        assert r.call_sites[0].parse_policy == "retry[3]"

    def test_ask_on_parse_error_text_warns(self) -> None:
        r = accept_type('ask("Q", on_parse_error: Abort())')
        assert len(r.warnings) == 1
        assert "on_parse_error" in r.warnings[0].message

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
        err = reject_type('let x = ask("Q", format: "cbor")\nx')
        assert "cbor" in str(err) or "codec" in str(err).lower()

    def test_ask_codec_mismatch_raises(self) -> None:
        # text codec doesn't support int kind
        err = reject_type('let n: int = ask("Q", format: "text")\nn')
        assert "text" in str(err) or "codec" in str(err).lower() or "support" in str(err).lower()

    def test_ask_strict_json_false(self) -> None:
        r = accept_type('let n: int = ask("Q", format: "json", strict_json: false)\nn')
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
        ('format: "text"', "strict_json: true", "on_parse_error: Abort()"),
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

    def test_unknown_named_arg_raises(self) -> None:
        err = reject_type('ask-request::[text]("Q", bogus: 1)')
        assert "bogus" in str(err) or "argument" in str(err).lower()

    def test_function_target_rejected(self) -> None:
        err = reject_type('ask-request::[(int) -> int]("Q")')
        assert "function" in str(err).lower() or "agent" in str(err).lower()

    def test_agent_target_rejected(self) -> None:
        err = reject_type('ask-request::[agent]("Q")')
        assert "function" in str(err).lower() or "agent" in str(err).lower()

    def test_with_explicit_agent(self) -> None:
        r = accept_type('agent reviewer\nask-request::[text]("Q", agent: reviewer)')
        assert r.resolved.program is not None

    def test_wrong_agent_type_raises(self) -> None:
        err = reject_type('let x = "no"\nask-request::[text]("Q", agent: x)')
        assert "agent" in str(err).lower()

    def test_strict_json_option(self) -> None:
        r = accept_type('ask-request::[int]("Q", format: "json", strict_json: true)')
        call = r.resolved.program.body.items[0]
        assert isinstance(call, Call)
        spec = r.contract_specs[call.node_id]
        assert spec.strict_json is True

    def test_on_parse_error_policy_recorded(self) -> None:
        r = accept_type('ask-request::[int]("Q", on_parse_error: Retry(n: 3))')
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
        assert "NoSuchType" in str(err) or "type" in str(err).lower()

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
        r = accept_type('let n: int = exec("ls", format: "json")\nn')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        spec = r.contract_specs[decl.value.node_id]
        assert spec.codec_name == "json"

    def test_exec_strict_json(self) -> None:
        r = accept_type('let n: int = exec("ls", format: "json", strict_json: true)\nn')
        decl = r.resolved.program.body.items[0]
        assert isinstance(decl, LetDecl)
        spec = r.contract_specs[decl.value.node_id]
        assert spec.strict_json is True

    def test_exec_on_parse_error_text_warns(self) -> None:
        r = accept_type('let x: text = exec("ls", on_parse_error: Abort())\nx')
        assert len(r.warnings) == 1
        assert "on_parse_error" in r.warnings[0].message

    def test_exec_call_site_record(self) -> None:
        r = accept_type('exec("ls")')
        assert len(r.call_sites) == 1
        cs = r.call_sites[0]
        assert cs.callee == "exec"

    def test_exec_call_site_abort_policy(self) -> None:
        r = accept_type('let n: int = exec("ls", on_parse_error: Abort())\nn')
        assert r.call_sites[0].parse_policy == "abort"

    def test_exec_call_site_retry_policy(self) -> None:
        r = accept_type('let n: int = exec("ls", on_parse_error: Retry(n: 2))\nn')
        assert r.call_sites[0].parse_policy == "retry[2]"

    def test_exec_strict_json_without_json_raises(self) -> None:
        err = reject_type('let x: text = exec("ls", strict_json: true)\nx')
        assert "strict_json" in str(err).lower() or "json" in str(err).lower()

    def test_exec_format_non_string_raises(self) -> None:
        err = reject_type('let n: int = exec("ls", format: 42)\nn')
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

    def test_funcdef_with_default_param(self) -> None:
        r = accept_type("def f(x: int, y: int = 0) -> int = x + y\nf(1)")
        assert "f" in r.function_signatures
        sig = r.function_signatures["f"]
        assert sig.params[1][2] is True  # has_default

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
        r = accept_type('def f(x: int) -> text = raise Abort(message: "err")\nf(1)')
        assert r.resolved.program is not None

    def test_funcdef_called_with_named_args(self) -> None:
        r = accept_type("def f(x: int, y: int = 0) -> int = x + y\nf(1, y: 2)")
        assert r.resolved.program is not None

    def test_funcdef_missing_required_arg(self) -> None:
        err = reject_type("def f(x: int, y: int) -> int = x + y\nf(1)")
        assert "missing" in str(err).lower() or "required" in str(err).lower()

    def test_funcdef_too_many_positional_args(self) -> None:
        err = reject_type("def f(x: int) -> int = x\nf(1, 2)")
        assert "too many" in str(err).lower() or "argument" in str(err).lower()

    def test_funcdef_unknown_named_arg(self) -> None:
        err = reject_type("def f(x: int) -> int = x\nf(z: 1)")
        assert "unknown" in str(err).lower() or "parameter" in str(err).lower()

    def test_funcdef_param_supplied_positionally_and_by_name(self) -> None:
        err = reject_type("def f(x: int) -> int = x\nf(1, x: 2)")
        assert "positionally" in str(err).lower() or "both" in str(err).lower()

    def test_funcdef_default_wrong_type_raises(self) -> None:
        err = reject_type('def f(x: int = "bad") -> int = x')
        assert "mismatch" in str(err).lower() or "expected" in str(err).lower()

    def test_funcdef_all_named_args(self) -> None:
        r = accept_type("def f(x: int, y: int) -> int = x + y\nf(x: 1, y: 2)")
        assert r.resolved.program is not None

    def test_funcdef_named_then_positional(self) -> None:
        # Named args after positionals — checker must handle ordering
        r = accept_type("def f(x: int, y: int = 0) -> int = x + y\nf(1)")
        assert r.resolved.program is not None

    def test_funcdef_duplicate_named_arg_via_call(self) -> None:
        # The parser catches duplicate args at parse time (AglSyntaxError)
        # so we test via reject_any instead of reject_type
        err = reject_any("def f(x: int, y: int) -> int = x\nf(y: 1, y: 2)")
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
        err = reject_type('fn() => raise Abort(message: "x")')
        assert "infer" in str(err).lower() or "return" in str(err).lower()

    def test_lambda_with_raise_and_annotation(self) -> None:
        r = accept_type('fn() -> int => raise Abort(message: "x")')
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
        err = reject_type("let f = fn(x: int) -> int => x\nf(x: 42)")
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
            "let res = Ok(value: 42)\n"
            "case res of | Result.Ok(value: v) => v | Result.Err(msg: m) => 0"
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
        r = accept_type('raise Abort(message: "error")')
        raise_node = r.resolved.program.body.items[0]
        assert isinstance(raise_node, Raise)
        t = r.node_types[raise_node.node_id]
        assert isinstance(t, BottomType)

    def test_raise_non_exception_rejected(self) -> None:
        err = reject_type("raise 42")
        assert "exception" in str(err).lower() or "raise" in str(err).lower()

    def test_raise_bottom_assignable_to_any(self) -> None:
        # raise can be used where any type is expected (annotated binding)
        r = accept_type('let x: int = raise Abort(message: "err")\nx')
        assert r.resolved.program is not None

    def test_raise_in_funcdef_body(self) -> None:
        r = accept_type('def f() -> text = raise Abort(message: "err")\nf()')
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

    def test_interpolated_function_rejected(self) -> None:
        err = reject_type('let f = fn(x: int) -> int => x\n"${f}"')
        assert "function" in str(err).lower() or "rendering" in str(err).lower()

    def test_interpolated_agent_rejected(self) -> None:
        err = reject_type('agent a\n"${a}"')
        assert "agent" in str(err).lower() or "rendering" in str(err).lower()

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
        err = reject_type('record R\n  x: int\nlet r = R(x: 1)\n"${{"a": r}}"')
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
        r = accept_type("1 = 2")
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
        err = reject_type('1 = "hello"')
        assert "same" in str(err).lower() or "equality" in str(err).lower()

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
        r = accept_type("record Point\n  x: int\n  y: int\nlet p = Point(x: 1, y: 2)\np.x")
        assert r.resolved.program is not None

    def test_record_unknown_field_raises(self) -> None:
        err = reject_type("record Point\n  x: int\nlet p = Point(x: 1)\np.z")
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
        assert "variant" in str(err).lower() or "Gone" in str(err)

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
        r = accept_type("record Point\n  x: int\n  y: int\nPoint(x: 1, y: 2)")
        assert r.resolved.program is not None

    def test_record_missing_field_raises(self) -> None:
        err = reject_type("record Point\n  x: int\n  y: int\nPoint(x: 1)")
        assert "missing" in str(err).lower() or "field" in str(err).lower()

    def test_record_unknown_field_raises(self) -> None:
        err = reject_type("record Point\n  x: int\nPoint(x: 1, z: 2)")
        assert "no field" in str(err).lower() or "field" in str(err).lower()

    def test_record_duplicate_arg_raises(self) -> None:
        # Parser catches duplicate field args
        err = reject_any("record Point\n  x: int\nPoint(x: 1, x: 2)")
        assert "duplicate" in str(err).lower() or "x" in str(err)

    def test_record_field_type_mismatch(self) -> None:
        err = reject_type('record Point\n  x: int\nPoint(x: "hello")')
        assert "mismatch" in str(err).lower() or "expected" in str(err).lower()

    def test_enum_variant_qualified(self) -> None:
        r = accept_type("enum Status\n  | Pass\n  | Fail\nStatus.Pass()")
        assert r.resolved.program is not None

    def test_enum_variant_unqualified_unique(self) -> None:
        r = accept_type("enum Status\n  | Pass\n  | Fail\nPass()")
        assert r.resolved.program is not None

    def test_enum_variant_ambiguous_raises(self) -> None:
        err = reject_type("enum A\n  | Pass\nenum B\n  | Pass\nPass()")
        assert "ambiguous" in str(err).lower() or "Pass" in str(err)

    def test_enum_variant_unknown_raises(self) -> None:
        err = reject_type("enum Status\n  | Pass\nStatus.Gone()")
        assert "variant" in str(err).lower() or "Gone" in str(err)

    def test_exception_constructor(self) -> None:
        r = accept_type('Abort(message: "error")')
        assert r.resolved.program is not None

    def test_abstract_exception_not_constructible(self) -> None:
        err = reject_type('Exception(message: "e")')
        assert "abstract" in str(err).lower() or "constructible" in str(err).lower()

    def test_unknown_constructor_raises(self) -> None:
        err = reject_type("Unknown(x: 1)")
        assert "unknown" in str(err).lower() or "Unknown" in str(err)

    def test_enum_variant_with_fields(self) -> None:
        # enum variants can have named fields
        r = accept_type(
            "enum Result\n  | Ok(value: int)\n  | Err(msg: text)\nOk(value: 42)"
        )
        assert r.resolved.program is not None

    def test_qualified_constructor_wrong_enum_raises(self) -> None:
        err = reject_type("enum A\n  | X\nenum B\n  | Y\nA.Y()")
        assert "variant" in str(err).lower() or "Y" in str(err)

    def test_qualified_constructor_not_enum_raises(self) -> None:
        err = reject_type("record R\n  x: int\nR.Something()")
        assert "enum" in str(err).lower() or "R" in str(err)


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
        err = reject_type("record R\n  x: int\nlet xs: list[json] = [R(x: 1)]\nxs")
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
        r = accept_type("record Point\n  x: int\n  y: int\nPoint(x: 1, y: 2)")
        assert r.resolved.program is not None

    def test_enum_def(self) -> None:
        r = accept_type("enum Color\n  | Red\n  | Blue\nRed()")
        assert r.resolved.program is not None

    def test_type_alias(self) -> None:
        r = accept_type("type MyInt = int\nlet x: MyInt = 42\nx")
        assert r.resolved.program is not None

    def test_duplicate_type_name_raises(self) -> None:
        err = reject_type("record A\n  x: int\nrecord A\n  y: int\nA(x: 1)")
        assert "already declared" in str(err).lower() or "duplicate" in str(err).lower()

    def test_record_recursive_raises(self) -> None:
        err = reject_type("record Node\n  child: Node\nNode(child: Node(child: ()))")
        assert "recursive" in str(err).lower()

    def test_enum_recursive_raises(self) -> None:
        err = reject_type(
            "enum List\n  | Cons(value: int, rest: List)\n  | Nil\nNil()"
        )
        assert "recursive" in str(err).lower()

    def test_record_duplicate_field_raises(self) -> None:
        err = reject_type("record R\n  x: int\n  x: text\nR(x: 1)")
        assert "duplicate" in str(err).lower() or "field" in str(err).lower()

    def test_enum_duplicate_variant_raises(self) -> None:
        err = reject_type("enum E\n  | A\n  | A\nA()")
        assert "duplicate" in str(err).lower() or "variant" in str(err).lower()

    def test_alias_cycle_raises(self) -> None:
        err = reject_type("type A = B\ntype B = A\n1")
        assert "cycle" in str(err).lower()

    def test_record_not_json_shaped(self) -> None:
        err = reject_type("record R\n  x: int\nlet r: json = R(x: 1)\nr")
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
        err = reject_type('var x = raise Abort(message: "e")\nx')
        assert "infer" in str(err).lower() or "raises" in str(err).lower()

    def test_let_bottom_inference_raises(self) -> None:
        err = reject_type('let x = raise Abort(message: "e")\nx')
        assert "infer" in str(err).lower() or "raises" in str(err).lower()

    def test_var_with_annotation_allows_bottom(self) -> None:
        r = accept_type('var x: int = raise Abort(message: "e")\nx')
        assert r.resolved.program is not None


# ---------------------------------------------------------------------------
# ParsePolicy constructors
# ---------------------------------------------------------------------------


class TestParsePolicy:
    def test_on_parse_error_abort(self) -> None:
        r = accept_type('let n: int = ask("Q", on_parse_error: Abort())\nn')
        assert r.call_sites[0].parse_policy == "abort"

    def test_on_parse_error_retry(self) -> None:
        r = accept_type('let n: int = ask("Q", on_parse_error: Retry(n: 5))\nn')
        assert r.call_sites[0].parse_policy == "retry[5]"

    def test_on_parse_error_invalid_constructor_raises(self) -> None:
        err = reject_type('let n: int = ask("Q", on_parse_error: 42)\nn')
        assert "on_parse_error" in str(err).lower() or "ParsePolicy" in str(err)

    def test_on_parse_error_abort_with_extra_args_raises(self) -> None:
        err = reject_type('let n: int = ask("Q", on_parse_error: Abort(message: "x"))\nn')
        assert "on_parse_error" in str(err).lower() or "Abort" in str(err)

    def test_on_parse_error_retry_no_n_raises(self) -> None:
        err = reject_type('let n: int = ask("Q", on_parse_error: Retry())\nn')
        assert "on_parse_error" in str(err).lower() or "Retry" in str(err)

    def test_on_parse_error_wrong_qualifier_raises(self) -> None:
        err = reject_type('let n: int = ask("Q", on_parse_error: Other.Abort())\nn')
        assert "on_parse_error" in str(err).lower() or "ParsePolicy" in str(err)

    def test_on_parse_error_text_target_warns(self) -> None:
        r = accept_type('ask("Q", on_parse_error: Abort())')
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
            sig.params = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Config pragma (pass-through)
# ---------------------------------------------------------------------------


class TestConfigPragma:
    def test_config_pragma_accepted(self) -> None:
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
        err = reject_type('let n: int = ask("Q", format: "text")\nn')
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
        err = reject_type('param x = raise Abort(message: "e")\nx')
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
        assert sig.params[0][2] is False   # x: not defaulted
        assert sig.params[1][2] is True    # y: has default

    def test_ask_strict_json_false(self) -> None:
        r = accept_type('let n: int = ask("Q", format: "json", strict_json: false)\nn')
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
            "enum E\n  | A(x: int)\nlet e = A(x: 1)\n"
            "case e of | E.A(x: n, x: m) => n | _ => 0"
        )
        assert "duplicate" in str(err).lower() or "field" in str(err).lower()

    def test_constructor_pattern_unknown_field_raises(self) -> None:
        err = reject_type(
            "enum E\n  | A(x: int)\nlet e = A(x: 1)\n"
            "case e of | E.A(z: n) => n | _ => 0"
        )
        assert "no field" in str(err).lower() or "z" in str(err)

    def test_variant_qualifier_wrong_raises(self) -> None:
        err = reject_type(
            "enum A\n  | X\nenum B\n  | X\nlet a = A.X()\na is B.X"
        )
        assert "qualifier" in str(err).lower() or "enum" in str(err).lower()

    def test_qualified_constructor_wrong_enum_raises(self) -> None:
        err = reject_type("enum A\n  | X\nenum B\n  | Y\nA.Y()")
        assert "variant" in str(err).lower() or "Y" in str(err)

    def test_qualified_constructor_not_enum_raises(self) -> None:
        err = reject_type("record R\n  x: int\nR.Something()")
        assert "enum" in str(err).lower() or "R" in str(err)

    def test_enum_variant_with_fields(self) -> None:
        r = accept_type(
            "enum Result\n  | Ok(value: int)\n  | Err(msg: text)\nOk(value: 42)"
        )
        assert r.resolved.program is not None

    def test_enum_variant_with_fields_case(self) -> None:
        r = accept_type(
            "enum Result\n  | Ok(value: int)\n  | Err(msg: text)\n"
            "let res = Ok(value: 42)\n"
            "case res of | Result.Ok(value: v) => v | Result.Err(msg: m) => 0"
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
        r = accept_type("record P\n  x: int\n  y: decimal\nP(x: 1, y: 2.5)")
        assert r.resolved.program is not None

    def test_builtin_type_name_shadow_raises(self) -> None:
        # ExecResult is a BUILTIN_PRELUDE_TYPE_NAMES — record shadows it
        err = reject_type("record ExecResult\n  x: int\nExecResult(x: 1)")
        assert "built-in" in str(err).lower() or "ExecResult" in str(err)

    def test_alias_to_record_field(self) -> None:
        # Exercises alias->record resolution in _ensure_referenced_type_built
        r = accept_type("record R\n  x: int\ntype MyR = R\nlet r: MyR = R(x: 1)\nr")
        assert r.resolved.program is not None

    def test_field_list_of_record(self) -> None:
        # Exercises ListT path in _ensure_referenced_type_built
        r = accept_type(
            "record R\n  x: int\nrecord S\n  items: list[R]\nS(items: [])"
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
        # Exercises line 1323->1325: qualifier check path
        r = accept_type("enum E\n  | A\n  | B\nlet e = E.A()\ne is E.A")
        assert r.resolved.program is not None

    def test_is_test_qualifier_not_enum_raises(self) -> None:
        # Exercises line 1337: qualifier resolves to non-enum type
        err = reject_type("enum A\n  | X\nrecord R\n  x: int\nlet a = A.X()\na is R.X")
        assert "not a known enum" in str(err).lower() or "enum" in str(err).lower()

    def test_is_test_unknown_qualifier_raises(self) -> None:
        # Exercises line 1337 path: qualifier name not registered as enum
        err = reject_type("enum E\n  | A\nlet e = E.A()\ne is UnknownEnum.A")
        assert "not a known enum" in str(err).lower() or "enum" in str(err).lower()

    def test_enum_variant_field_duplicate_raises(self) -> None:
        # Exercises line 287: duplicate field in enum variant
        err = reject_type("enum E\n  | A(x: int, x: text)\nA(x: 1)")
        assert "duplicate" in str(err).lower() or "field" in str(err).lower()

    def test_unqualified_ctor_with_enum_expected_type(self) -> None:
        # Exercises line 1409-1410: expected EnumType candidate disambiguation
        r = accept_type("enum E\n  | A\n  | B\nlet x: E = A()\nx")
        assert r.resolved.program is not None

    def test_branch_decimal_int_widening(self) -> None:
        # if true => 2.5 | else => 2 → decimal (exercises 1663: decimal+int)
        r = accept_type("if true => 2.5 | true => 3 | else => 1.0")
        if_node = r.resolved.program.body.items[0]
        assert isinstance(if_node, If)
        t = r.node_types[if_node.node_id]
        assert t == DecimalType()

    def test_constructor_pattern_with_qualifier(self) -> None:
        # Exercises line 1586->1588: ctor pattern qualifier check
        r = accept_type(
            "enum E\n  | A(x: int)\nlet e = A(x: 1)\n"
            "case e of | E.A(x: n) => n | _ => 0"
        )
        assert r.resolved.program is not None

    def test_constructor_pattern_wrong_variant_raises(self) -> None:
        # Exercises line 1590: ctor pattern variant not found
        err = reject_type(
            "enum E\n  | A\n  | B\nlet e = A()\n"
            "case e of | E.C() => 1 | _ => 0"
        )
        assert "variant" in str(err).lower() or "C" in str(err)

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
        err = reject_type('let n: int = ask("Q", on_parse_error: Retry(n: "bad"))\nn')
        assert "on_parse_error" in str(err).lower() or "Retry" in str(err)

    def test_retry_with_wrong_key_raises(self) -> None:
        # Exercises line 880 -> falls through to raise
        err = reject_type('let n: int = ask("Q", on_parse_error: Retry(m: 3))\nn')
        assert "on_parse_error" in str(err).lower() or "Retry" in str(err)

    def test_parse_policy_unknown_variant_raises(self) -> None:
        # Exercises line 877->890: arg.name is neither "Abort" nor "Retry"
        err = reject_type('let n: int = ask("Q", on_parse_error: ParsePolicy.Bad())\nn')
        assert "on_parse_error" in str(err).lower() or "ParsePolicy" in str(err)

    def test_exec_strict_json_non_bool_raises(self) -> None:
        # Exercises line 815: strict_json non-BoolLit in exec
        err = reject_type('let n: int = exec("ls", format: "json", strict_json: "yes")\nn')
        assert "strict_json" in str(err).lower() or "bool" in str(err).lower()

    def test_decimal_subtraction_yields_decimal(self) -> None:
        # Exercises line 1282: _check_numeric_binop returns DecimalType
        r = accept_type("1.5 - 0.5")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == DecimalType()

    def test_decimal_multiplication_yields_decimal(self) -> None:
        # Also exercises line 1282
        r = accept_type("2.0 * 3.0")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == DecimalType()

    def test_list_decimal_then_int_widening(self) -> None:
        # Exercises line 1550->1547: unified=decimal, t=int, is_assignable(dec,int)=False
        # but is_assignable(int,decimal)=True so we continue
        r = accept_type("[2.5, 1]")
        node = r.resolved.program.body.items[0]
        assert r.node_types[node.node_id] == ListType(elem=DecimalType())

    def test_alias_field_in_record(self) -> None:
        # Exercises lines 314-316: alias in _ensure_referenced_type_built
        r = accept_type("type N = int\nrecord R\n  x: N\nR(x: 1)")
        assert r.resolved.program is not None

    def test_list_field_of_aliased_type(self) -> None:
        # Exercises line 321: ListT in _ensure_referenced_type_built
        r = accept_type("type N = int\nrecord R\n  xs: list[N]\nR(xs: [])")
        assert r.resolved.program is not None

    def test_dict_field_of_aliased_type(self) -> None:
        # Exercises line 323: DictT in _ensure_referenced_type_built
        r = accept_type("type N = int\nrecord R\n  d: dict[text, N]\nR(d: {})")
        assert r.resolved.program is not None

    def test_two_records_same_enum_field(self) -> None:
        # Exercises line 246: _ensure_built_enum called twice returns early
        r = accept_type("enum E\n  | A\nrecord R1\n  e: E\nrecord R2\n  e: E\nR1(e: A())")
        assert r.resolved.program is not None

    def test_template_nested_list_in_list(self) -> None:
        # Exercises lines 1165-1167: non-empty ListLit as child of template list
        r = accept_type('"${[1, [2, 3]]}"')
        assert r.resolved.program is not None

    def test_template_nested_dict_in_dict(self) -> None:
        # Exercises lines 1171-1173: non-empty DictLit as child of template dict
        r = accept_type('"${{"a": {"b": 1}}}"')
        assert r.resolved.program is not None

    def test_is_test_without_qualifier(self) -> None:
        # Exercises line 1323->1325: qualifier is None, skip qualifier check
        r = accept_type("enum E\n  | A\n  | B\nlet e = E.A()\ne is A")
        assert r.resolved.program is not None

    def test_constructor_pattern_without_qualifier(self) -> None:
        # Exercises line 1586->1588: pattern qualifier is None, skip qualifier check
        r = accept_type(
            "enum E\n  | A(x: int)\nlet e = A(x: 1)\n"
            "case e of | A(x: n) => n | _ => 0"
        )
        assert r.resolved.program is not None

    def test_record_field_of_builtin_record_type(self) -> None:
        # Exercises 314->exit: NameT("ExecResult") is in env but not in _record_defs
        # (it's a built-in type, not user-declared)
        r = accept_type(
            "record Wrapper\n  result: ExecResult\n"
            "let x = exec(\"ls\")\nWrapper(result: x)"
        )
        assert r.resolved.program is not None

    def test_funcdef_builtin_type_name_rejected(self) -> None:
        # Exercises line 367: def named after a built-in type is rejected by the typechecker
        # (scope does not reject 'text'/'int'/etc. as def names, only print/exec/ask)
        err = reject_type("def text() -> int = 1\ntext()")
        assert "built-in type name" in str(err)

    def test_all_bottom_if_branches_yield_bottom(self) -> None:
        # Exercises line 1655: _unify_branch_types returns BottomType when all branches
        # are BottomType (i.e. all branches always raise)
        r = accept_type(
            "def f(x: int) -> int =\n"
            "  if x = 0 =>\n"
            "    let msg = \"zero\"\n"
            "    raise Abort(message: msg)\n"
            "  | else =>\n"
            "    let msg = \"nonzero\"\n"
            "    raise Abort(message: msg)"
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
        # Exercises line 424: _check_block returns UnitType() for an empty block.
        # The grammar never produces an empty block from source, so we construct
        # the AST directly.
        sp = mk_span()
        block = Block(items=(), span=sp, node_id=_mk_node_id())
        prog = Program(body=block, span=sp, node_id=_mk_node_id())
        resolved = self._mk_resolved(prog)
        result = check(resolved, default_capabilities())
        assert result is not None

    def test_empty_case_branches_fallback(self) -> None:
        # Exercises line 1067: _check_case returns fallback type when branches is empty.
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
        # Exercises line 1449: duplicate constructor argument check.
        # The parser rejects duplicates, so we construct the node directly.
        sp = mk_span()
        val1 = IntLit(value=1, span=sp, node_id=_mk_node_id())
        val2 = IntLit(value=2, span=sp, node_id=_mk_node_id())
        na1 = NamedArg(name="message", value=val1, span=sp, node_id=_mk_node_id())
        na2 = NamedArg(name="message", value=val2, span=sp, node_id=_mk_node_id())
        ctor = Constructor(
            qualifier=None, name="Abort", args=(na1, na2), span=sp, node_id=_mk_node_id()
        )
        block = Block(items=(ctor,), span=sp, node_id=_mk_node_id())
        prog = Program(body=block, span=sp, node_id=_mk_node_id())
        resolved = self._mk_resolved(prog)
        with pytest.raises(AglTypeError, match="Duplicate argument"):
            check(resolved, default_capabilities())

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
        # Exercises line 935: _check_declared_name_call falls back to value-call
        # when get_function_signature returns None.  This happens when a FuncDef
        # is in declared_functions but not in the block items (so the pre-pass skips it).
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
        # h IS in declared_functions → _check_declared_name_call runs, sig is None,
        # falls through to value-call → binding type of h not set → AssertionError
        resolved = self._mk_resolved(
            prog,
            resolution={callee_nid: binding_ref},
            declared_functions={"h": fd},
        )
        with pytest.raises(AssertionError, match="checker invariant"):
            check(resolved, default_capabilities())

    def test_duplicate_named_arg_in_declared_call(self) -> None:
        # Exercises line 970: duplicate named arg check in _check_declared_name_call.
        # The parser rejects duplicate named args, so we construct directly.
        sp = mk_span()
        p_nid = _mk_node_id()
        ret_t = IntT(span=sp, node_id=_mk_node_id())
        param_t = IntT(span=sp, node_id=_mk_node_id())
        param = Param(name="x", type_expr=param_t, default=None, span=sp, node_id=p_nid)
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
        with pytest.raises(AglTypeError, match="Duplicate named argument"):
            check(resolved, default_capabilities())


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
            "  classify(y: 5)\n"
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
            "add(1, y: 2)"
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
            "def apply(f: (int) -> int, x: int) -> int = f(n: x)\n"
            "apply(fn(n: int) => n, 7)"
        )
        assert "named" in str(err).lower() or "declared" in str(err).lower()

    def test_field_access_callee_is_value_call(self) -> None:
        # A field that holds a function value must be treated as a value call.
        r = accept_type(
            "record Wrapper\n"
            "  fn_field: (int) -> int\n"
            "let w = Wrapper(fn_field: fn(x: int) -> int => x)\n"
            "w.fn_field(42)"
        )
        assert r.resolved.program is not None


# ---------------------------------------------------------------------------
# Fix 2: ask/exec reject unknown named args and extra positionals
# ---------------------------------------------------------------------------


class TestAskUnknownArgs:
    def test_ask_unknown_named_arg_rejected(self) -> None:
        err = reject_type('ask("Q", bogus: 1)')
        assert "ask" in str(err).lower() or "unknown" in str(err).lower()
        assert "bogus" in str(err)

    def test_ask_typo_named_arg_rejected(self) -> None:
        err = reject_type('ask("Q", strict_jsonn: true)')
        assert "ask" in str(err).lower() or "unknown" in str(err).lower()
        assert "strict_jsonn" in str(err)

    def test_ask_extra_positional_rejected(self) -> None:
        err = reject_type('ask("Q", "extra")')
        assert "ask" in str(err).lower() or "positional" in str(err).lower()

    def test_ask_valid_named_arg_combinations_still_accepted(self) -> None:
        # All four known named args together must be accepted.
        r = accept_type(
            'agent a\nlet n: int = ask("Q", agent: a, format: "json",'
            ' strict_json: true, on_parse_error: Abort())\nn'
        )
        assert r.resolved.program is not None


class TestExecUnknownArgs:
    def test_exec_unknown_named_arg_rejected(self) -> None:
        err = reject_type('exec("ls", bogus: 1)')
        assert "exec" in str(err).lower() or "unknown" in str(err).lower()
        assert "bogus" in str(err)

    def test_exec_agent_named_arg_rejected(self) -> None:
        # exec has no 'agent:' argument (D10).
        err = reject_type('agent a\nexec("ls", agent: a)')
        assert "exec" in str(err).lower() or "unknown" in str(err).lower()

    def test_exec_extra_positional_rejected(self) -> None:
        err = reject_type('exec("ls", "extra")')
        assert "exec" in str(err).lower() or "positional" in str(err).lower()

    def test_exec_valid_named_arg_combinations_still_accepted(self) -> None:
        # format, strict_json, on_parse_error are valid for exec.
        r = accept_type(
            'let n: int = exec("ls", format: "json", strict_json: true,'
            ' on_parse_error: Abort())\nn'
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
        err = reject_type('exec("ls", format: "json")')
        assert "ExecResult" in str(err) or "format" in str(err).lower()

    def test_exec_structured_strict_json_rejected(self) -> None:
        err = reject_type('exec("ls", strict_json: true)')
        assert "ExecResult" in str(err) or "strict_json" in str(err).lower()

    def test_exec_structured_on_parse_error_rejected(self) -> None:
        err = reject_type('exec("ls", on_parse_error: Abort())')
        assert "ExecResult" in str(err) or "on_parse_error" in str(err).lower()

    def test_exec_parsed_form_has_no_structured_exec(self) -> None:
        r = accept_type('let n: int = exec("ls", format: "json")\nn')
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
            "let b = Box(compute: fn(n: int) -> int => n * 2)\n"
            "b.compute(5)"
        )
        assert r.resolved.program is not None

    def test_field_access_function_callee_named_arg_rejected(self) -> None:
        # Named arg on a field-access callee is rejected.
        err = reject_type(
            "record Box\n"
            "  compute: (int) -> int\n"
            "let b = Box(compute: fn(n: int) -> int => n * 2)\n"
            "b.compute(n: 5)"
        )
        assert "named" in str(err).lower() or "declared" in str(err).lower()

    def test_exec_function_target_rejected(self) -> None:
        # exec into a function/agent type is a static error.
        err = reject_type('let f: (int) -> int = exec("ls")\nf(1)')
        assert "function" in str(err).lower() or "agent" in str(err).lower()


# ---------------------------------------------------------------------------
# Cast typecheck tests (M3)
# ---------------------------------------------------------------------------


class TestCast:
    """Tests for the Cast node type checking."""

    def test_int_as_text_total_render(self) -> None:
        """int as text yields text."""
        r = accept_type("let s: text = 1 as text\ns")
        assert r  # no exception

    def test_text_as_json_total_json(self) -> None:
        """text as json yields json."""
        r = accept_type('let j: json = "hello" as json\nj')
        assert r

    def test_text_as_int_fallible_yields_int(self) -> None:
        """text as int yields int type."""
        r = accept_type('let x: int = "42" as int\nx')
        assert r

    def test_decimal_as_int_fallible(self) -> None:
        """decimal as int is fallible, yields int."""
        r = accept_type("let d = 3.5\nlet x: int = d as int\nx")
        assert r

    def test_int_as_decimal_noop(self) -> None:
        """int as decimal is a no-op widen."""
        r = accept_type("let d: decimal = 1 as decimal\nd")
        assert r

    def test_as_question_yields_bool(self) -> None:
        """as? always yields bool."""
        r = accept_type('let b: bool = "42" as? int\nb')
        assert r

    def test_as_question_on_total_cast_yields_bool(self) -> None:
        """as? on a total cast also yields bool."""
        r = accept_type("let b: bool = 1 as? text\nb")
        assert r

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

    def test_record_as_json_rejected(self) -> None:
        """record as json is a static error."""
        err = reject_type("record R\n  x: int\nlet r = R(x: 1)\nr as json")
        assert "cannot cast" in str(err).lower()

    def test_cast_to_unit_rejected(self) -> None:
        """casting to unit is a static error."""
        err = reject_type("1 as unit")
        assert "cannot cast" in str(err).lower() or "unit" in str(err).lower()

    def test_as_question_same_static_error(self) -> None:
        """as? on a static-error pair is also a static error."""
        err = reject_type("true as? int")
        assert "cannot cast" in str(err).lower()

    def test_json_to_text_render(self) -> None:
        """json as text yields text (TOTAL_RENDER — D1 completeness)."""
        r = accept_type('let j: json = 42\nlet s: text = j as text\ns')
        assert r

    def test_json_as_text_let_binding(self) -> None:
        """json as text typechecks: let s: text = (1 as json) as text."""
        r = accept_type("let s: text = (1 as json) as text\ns")
        assert r

    def test_chained_cast_int_to_json_to_text(self) -> None:
        """Chained x as json as text resolves to text (D5 example)."""
        r = accept_type("let x: int = 1\nlet s: text = x as json as text\ns")
        assert r

    def test_json_to_list_fallible(self) -> None:
        """json as list[int] yields list[int]."""
        r = accept_type("let j: json = 42\nlet xs: list[int] = j as list[int]\nxs")
        assert r

    def test_cast_spec_stored(self) -> None:
        """CastSpec is stored in CheckedProgram.cast_specs."""
        from agm.agl.typecheck.types import CastKind
        r = accept_type("1 as text")
        assert len(r.cast_specs) == 1
        spec = next(iter(r.cast_specs.values()))
        assert spec.kind == CastKind.TOTAL_RENDER

    def test_as_question_spec_stored(self) -> None:
        """as? CastSpec is stored with same kind as as."""
        from agm.agl.typecheck.types import CastKind
        r = accept_type('"hello" as? int')
        assert len(r.cast_specs) == 1
        spec = next(iter(r.cast_specs.values()))
        assert spec.kind == CastKind.FALLIBLE

    def test_raise_operand_can_be_cast(self) -> None:
        """A cast preserves the universal assignability of bottom."""
        r = accept_type('(raise Abort(message: "x")) as text')
        assert r


class TestParseJsonCall:
    """Tests for parse_json built-in."""

    def test_parse_json_returns_json(self) -> None:
        """parse_json("...") yields json."""
        r = accept_type('let j: json = parse_json("42")\nj')
        assert r

    def test_parse_json_named_arg_rejected(self) -> None:
        """Named args to parse_json are rejected."""
        err = reject_type('parse_json(text: "42")')
        assert "parse_json" in str(err).lower() or "positional" in str(err).lower()

    def test_parse_json_wrong_arity_rejected(self) -> None:
        """parse_json() with wrong arity is rejected."""
        err = reject_type('parse_json("a", "b")')
        assert "parse_json" in str(err).lower() or "one" in str(err).lower()

    def test_parse_json_no_args_rejected(self) -> None:
        """parse_json() with no args is rejected."""
        err = reject_type("parse_json()")
        assert "parse_json" in str(err).lower()
