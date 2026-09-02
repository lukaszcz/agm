"""Tests for the shared nominal type-declaration table.

Covers ``TypeDef``/``TypeTable`` in ``agm.agl.semantics.type_table`` — the
sole source of record/enum field and variant shapes, since ``RecordType``/
``EnumType`` handles carry none — and how it is populated: the type builder
(``typecheck/builder.py``), the graph pre-pass (``typecheck/graph.py``), and
REPL session accumulation (``TypeEnvironment.seed_from``).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.ir.reserved_nominals import (
    NO_DECL_ID,
    RESERVED_NOMINAL_NAMES,
    reserved_nominal_id,
)
from agm.agl.modules.ids import ENTRY_ID, RESERVED_ID, STD_PRELUDE_ID, ModuleId
from agm.agl.repl import ReplSession
from agm.agl.scope.program import resolve_program
from agm.agl.semantics.analyses import (
    compute_finite_closure,
    compute_uninhabited,
    nominal_references,
)
from agm.agl.semantics.type_table import (
    BUILTIN_PRELUDE_TYPE_DEFS,
    MethodDef,
    TypeDef,
    TypeTable,
    cast_classification,
    comparable_types,
    create_seeded_type_table,
    is_json_convertible,
    json_cast_hint,
    source_enum_member_decl_id,
)
from agm.agl.semantics.types import (
    BUILTIN_PRELUDE_TYPES,
    COMPATIBILITY_PRELUDE_TYPE_NAMES,
    ArrayType,
    BoolType,
    BottomType,
    CastKind,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    FunctionType,
    InferenceVarType,
    IntType,
    JsonType,
    RecordType,
    TextType,
    Type,
    TypeVarType,
    UnitType,
)
from agm.agl.syntax.nodes import (
    EnumDef,
    ExceptionDef,
    LetDecl,
    ParamKind,
    RecordDef,
    VarDecl,
    VariantDef,
    simple_let_pattern_name,
)
from agm.agl.typecheck import AglTypeError, CheckedModule
from agm.agl.typecheck.program import check_program
from tests._agl_helpers import enum_typedef, register_typedef, strip_decl_ids
from tests.agl.ir_harness import evaluate_ir_output, make_graph_from_files
from tests.agl.module_graph import resolve_and_check_inline_entry

_CAPS = HostCapabilities(
    supports_shell_exec=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}),
    },
)

_LIB_ID = ModuleId.from_path("lib")


def _check(src: str, *, default_stdlib: bool = True) -> CheckedModule:
    """Resolve + check *src* as the entry of a real module graph."""
    return resolve_and_check_inline_entry(src, _CAPS, default_stdlib=default_stdlib)


def test_builtin_member_identity_falls_back_for_non_enum_prelude_types() -> None:
    assert (
        source_enum_member_decl_id(
            STD_PRELUDE_ID,
            (),
            "ExecResult",
            "not-a-member",
            42,
            is_builtin=True,
        )
        == 42
    )


def test_enum_owner_for_member_recovers_only_captured_type_arguments() -> None:
    table = TypeTable()
    outcome = TypeDef(
        kind="enum",
        name="Outcome",
        module_id=ENTRY_ID,
        type_params=("T", "E"),
        members=(
            RecordType("ok", (TypeVarType("T"),), scope_path=("Outcome",), decl_id=1),
            RecordType("fixed", (IntType(),), scope_path=("Outcome",), decl_id=2),
        ),
        decl_node_id=2,
    )
    table.register(outcome)
    members = table.enum_members(outcome.handle((IntType(), TextType())))

    assert table.enum_owner_for_member(members[0]) is None
    assert table.enum_owner_for_member(members[1]) is None


def test_enum_owner_for_referenced_member_requires_its_full_type_template() -> None:
    table = TypeTable()
    box = TypeDef(
        kind="record",
        name="Box",
        module_id=ENTRY_ID,
        type_params=("T",),
        fields=(("value", TypeVarType("T")),),
        decl_node_id=10,
    )
    enum = TypeDef(
        kind="enum",
        name="E",
        module_id=ENTRY_ID,
        members=(RecordType("Box", (IntType(),), module_id=ENTRY_ID, decl_id=10),),
        decl_node_id=11,
    )
    table.register(box)
    table.register(enum)

    assert table.enum_owner_for_member(RecordType("Box", (IntType(),), decl_id=10)) == enum.handle()
    assert table.enum_owner_for_member(RecordType("Box", (TextType(),), decl_id=10)) is None
    assert not table.record_matches_enum_member(
        enum.handle(), "Missing", RecordType("Box", (IntType(),), decl_id=10)
    )


def test_enum_member_template_requires_the_same_record_declaration() -> None:
    enum = TypeDef(
        kind="enum",
        name="E",
        module_id=ENTRY_ID,
        members=(RecordType("R", module_id=ENTRY_ID, decl_id=10),),
        decl_node_id=11,
    )

    assert (
        TypeTable._match_enum_member_template(
            enum,
            enum.members[0],
            RecordType("R", module_id=ENTRY_ID, decl_id=12),
        )
        is None
    )


def test_referenced_member_with_concrete_arguments_does_not_share_enum_membership() -> None:
    table = TypeTable()
    table.register(
        TypeDef(
            kind="record",
            name="Box",
            module_id=ENTRY_ID,
            type_params=("T",),
            fields=(("value", TypeVarType("T")),),
            decl_node_id=10,
        )
    )
    table.register(
        TypeDef(
            kind="enum",
            name="E",
            module_id=ENTRY_ID,
            members=(RecordType("Box", (IntType(),), module_id=ENTRY_ID, decl_id=10),),
            decl_node_id=11,
        )
    )

    assert not table.records_share_enum_membership(
        (
            RecordType("Box", (IntType(),), decl_id=10),
            RecordType("Box", (TextType(),), decl_id=10),
        )
    )


def test_mixed_referenced_members_with_concrete_arguments_raise_a_type_error() -> None:
    with pytest.raises(AglTypeError):
        _check(
            "record Box[T](value: T)\n"
            "enum E = ::Box[int]\n"
            'let values = [Box(value = 1), Box(value = "text")]\n'
            "values"
        )


def test_scoped_generic_enum_does_not_claim_the_root_type_or_constructor_namespace() -> None:
    checked = _check(
        "enum A::Choice[T]\n"
        "  | hidden(value: T)\n"
        "enum Choice[T]\n"
        "  | visible(value: T)\n"
        "let result = visible(value = 1)\n"
        "result"
    )

    assert "Choice" in checked.type_env.all_generic_types()
    assert "hidden" not in checked.resolved.constructor_candidates


def test_type_references_inside_a_scope_use_the_nearest_scoped_type() -> None:
    checked = _check(
        "record T(value: text)\n"
        "\n"
        "scope A\n"
        "  record T(value: int)\n"
        "  def build(value: T) -> T = value\n"
        "end A\n"
        "\n"
        "A::build(A::T(value = 1))"
    )

    result_type = checked.node_types[checked.resolved.program.body.items[-1].node_id]
    assert strip_decl_ids(result_type) == RecordType("T", scope_path=("A",))


def _check_program(tmp_path: Path, modules: dict[str, str]):
    """Build and typecheck a multi-module graph; returns CheckedProgram."""
    graph = make_graph_from_files(tmp_path, modules)
    resolved_program = resolve_program(graph)
    return check_program(resolved_program, _CAPS)


def test_standalone_current_module_type_anchor_bypasses_nested_scope_during_lowering() -> None:
    assert (
        evaluate_ir_output(
            "record T(value: int)\n"
            "\n"
            "scope A\n"
            "  record T(value: text)\n"
            "  def root(value: ::T) -> ::T = value\n"
            "end A\n"
            "\n"
            "print(A::root(T(value = 1)).value)"
        )
        == "1\n"
    )


def test_scoped_aliases_are_available_to_scoped_function_signatures() -> None:
    checked = _check(
        "scope A\n"
        "  type Count = int\n"
        "  record Marker()\n"
        "  def keep(value: Count) -> Count = value\n"
        "end A\n"
        "\n"
        "A::keep(1)"
    )

    assert checked.node_types


def test_current_module_generic_type_anchor_rejects_unknown_root_type() -> None:
    with pytest.raises(AglTypeError, match="Unknown scoped type"):
        _check("def f(value: ::A::Missing[int]) -> int = 0\nf(1)")


def test_program_type_table_keys_keep_root_and_scoped_nominals_distinct(tmp_path: Path) -> None:
    checked = _check_program(
        tmp_path,
        {
            "entry": "record A::Visible(value: int)\nrecord Visible(value: text)\n()",
        },
    )

    assert strip_decl_ids(checked.program_type_table[(ENTRY_ID, (), "Visible")]) == RecordType(
        "Visible"
    )
    assert strip_decl_ids(checked.program_type_table[(ENTRY_ID, ("A",), "Visible")]) == RecordType(
        "Visible", scope_path=("A",)
    )


def test_scoped_type_context_restores_after_a_type_error(tmp_path: Path) -> None:
    source = "scope A\n  record Broken(value: Missing)\nend A"

    with pytest.raises(AglTypeError):
        _check(source)
    with pytest.raises(AglTypeError):
        _check_program(tmp_path, {"entry": source})


def test_scoped_generic_type_applications_resolve_in_module_and_program_contexts(
    tmp_path: Path,
) -> None:
    source = (
        "scope A\n"
        "  record G[T](value: T)\n"
        "  def keep(value: A::G[int]) -> A::G[int] = value\n"
        "end A\n"
        "\n"
        "A::G::[int](value = 1)"
    )

    assert _check(source).node_types
    assert _check_program(tmp_path, {"entry": source}).modules[ENTRY_ID].node_types


def _enum_fields(table: TypeTable, handle: EnumType) -> dict[str, dict[str, Type]]:
    """Project a member set into the legacy name-to-fields test view."""
    return {
        name: dict(table.record_fields(member))
        for name, member in table.enum_member_names(handle).items()
    }


def _binding_value_type(checked: CheckedModule, name: str):
    """Inferred type of the RHS of the top-level ``let``/``var <name> = ...``."""
    for item in checked.resolved.program.body.items:
        if isinstance(item, VarDecl) and item.name == name:
            return checked.node_types[item.value.node_id]
        if isinstance(item, LetDecl) and simple_let_pattern_name(item.pattern) == name:
            return checked.node_types[item.value.node_id]
    raise AssertionError(f"no top-level binding named {name!r}")


def test_inline_enum_members_are_scoped_records_with_captured_parameters() -> None:
    checked = _check("enum E[T, U]\n  | Leaf\n  | One(value: U)\n  | Both(left: T, right: U)\n()")

    enum_type = checked.type_env.get_generic_type("E")
    assert enum_type is not None
    members = checked.type_env.type_table.enum_members(
        EnumType("E", type_args=(IntType(), TextType()), decl_id=enum_type.template.decl_id)
    )
    assert [member.name for member in members] == ["Leaf", "One", "Both"]
    enum_def = next(
        item for item in checked.resolved.program.body.items if isinstance(item, EnumDef)
    )
    source_member_ids = [
        member.node_id for member in enum_def.members if isinstance(member, VariantDef)
    ]
    assert [member.decl_id for member in members] == source_member_ids
    assert [member.type_args for member in members] == [(), (TextType(),), (IntType(), TextType())]
    assert checked.type_env.type_table.enum_member_names(
        EnumType("E", type_args=(IntType(), TextType()), decl_id=enum_type.template.decl_id)
    ) == {"Leaf": members[0], "One": members[1], "Both": members[2]}
    member_defs = [checked.type_env.type_table.get_by_id(member.decl_id) for member in members]
    assert all(member_def is not None for member_def in member_defs)
    assert [member_def.type_params for member_def in member_defs if member_def is not None] == [
        (),
        ("U",),
        ("T", "U"),
    ]


def test_inline_member_aliases_capture_only_resolved_parameters() -> None:
    checked = _check(
        "type Ignore[T] = int\n"
        "enum E[T] | M(value: Ignore[T])\n"
        "let value: E::M = M(value = 1)\n"
        "value"
    )

    enum_type = checked.type_env.get_generic_type("E")
    assert enum_type is not None
    member = checked.type_env.type_table.enum_member_names(enum_type.template)["M"]
    member_def = checked.type_env.type_table.get_by_id(member.decl_id)

    assert member_def is not None
    assert member_def.type_params == ()
    assert _binding_value_type(checked, "value") == member


def test_forward_inline_member_uses_arity_after_alias_erasure() -> None:
    checked = _check(
        "record A(value: E::M)\n"
        "type Ignore[T] = int\n"
        "enum E[T] | M(value: Ignore[T])\n"
        "A(value = M(value = 1))"
    )

    value_type = checked.node_types[checked.resolved.program.body.items[-1].node_id]
    assert isinstance(value_type, RecordType)
    assert value_type.name == "A"


def test_forward_inline_member_rejects_type_arguments_erased_by_alias() -> None:
    with pytest.raises(AglTypeError):
        _check(
            "record A(value: E::M[int])\ntype Ignore[T] = int\nenum E[T] | M(value: Ignore[T])\n()"
        )


# ---------------------------------------------------------------------------
# register / get
# ---------------------------------------------------------------------------


class TestTypeDefHandle:
    def test_record_handle(self) -> None:
        typedef = TypeDef(kind="record", name="Point", module_id=ENTRY_ID, decl_node_id=700000)
        assert typedef.handle() == RecordType(name="Point", module_id=ENTRY_ID, decl_id=700000)

    def test_enum_handle(self) -> None:
        typedef = TypeDef(kind="enum", name="Color", module_id=ENTRY_ID, decl_node_id=700001)
        assert typedef.handle() == EnumType(name="Color", module_id=ENTRY_ID, decl_id=700001)

    def test_generic_record_handle_with_type_args(self) -> None:
        typedef = TypeDef(
            kind="record", name="Box", module_id=ENTRY_ID, type_params=("T",), decl_node_id=700002
        )
        handle = typedef.handle(type_args=(IntType(),))
        assert handle == RecordType(
            name="Box", type_args=(IntType(),), module_id=ENTRY_ID, decl_id=700002
        )

    def test_exception_handle(self) -> None:
        typedef = TypeDef(kind="exception", name="Boom", module_id=ENTRY_ID, decl_node_id=700003)
        assert typedef.handle() == ExceptionType(name="Boom", module_id=ENTRY_ID, decl_id=700003)

    def test_exception_handle_rejects_type_args(self) -> None:
        typedef = TypeDef(kind="exception", name="Boom", module_id=ENTRY_ID, decl_node_id=700003)
        with pytest.raises(ValueError, match="does not accept type_args"):
            typedef.handle(type_args=(IntType(),))

    def test_record_handle_stamps_decl_id_from_decl_node_id(self) -> None:
        typedef = TypeDef(kind="record", name="Point", module_id=ENTRY_ID, decl_node_id=42)
        assert typedef.handle().decl_id == 42

    def test_enum_handle_stamps_decl_id_from_decl_node_id(self) -> None:
        typedef = TypeDef(kind="enum", name="Color", module_id=ENTRY_ID, decl_node_id=43)
        assert typedef.handle().decl_id == 43

    def test_exception_handle_stamps_decl_id_from_decl_node_id(self) -> None:
        typedef = TypeDef(kind="exception", name="Boom", module_id=ENTRY_ID, decl_node_id=44)
        assert typedef.handle().decl_id == 44


class TestRegisterAndGet:
    def test_register_and_get_record_in_entry_module(self) -> None:
        table = TypeTable()
        typedef = TypeDef(
            kind="record",
            name="Point",
            module_id=ENTRY_ID,
            fields=(("x", IntType()), ("y", IntType())),
            decl_node_id=700000,
        )
        table.register(typedef)
        assert table.get(ENTRY_ID, "Point") == typedef

    def test_register_and_get_enum_in_non_entry_module(self) -> None:
        table = TypeTable()
        typedef = enum_typedef("Color", {"Red": {}, "Blue": {}}, module_id=_LIB_ID, decl_id=700004)
        register_typedef(table, typedef)
        found = table.get(_LIB_ID, "Color")
        assert found is not None
        assert found.kind == typedef.kind
        assert tuple(member.name for member in found.members) == ("Red", "Blue")

    def test_get_missing_key_returns_none(self) -> None:
        table = TypeTable()
        assert table.get(ENTRY_ID, "Nope") is None

    def test_same_name_in_different_modules_are_independent_entries(self) -> None:
        table = TypeTable()
        entry_def = TypeDef(
            kind="record",
            name="Widget",
            module_id=ENTRY_ID,
            fields=(("a", IntType()),),
            decl_node_id=700005,
        )
        lib_def = TypeDef(
            kind="record",
            name="Widget",
            module_id=_LIB_ID,
            fields=(("b", TextType()),),
            decl_node_id=700006,
        )
        table.register(entry_def)
        table.register(lib_def)
        assert table.get(ENTRY_ID, "Widget") == entry_def
        assert table.get(_LIB_ID, "Widget") == lib_def


# ---------------------------------------------------------------------------
# record_fields / enum_members on non-generic handles
# ---------------------------------------------------------------------------


class TestNonGenericAccessors:
    def test_record_fields_non_generic(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Point",
                module_id=ENTRY_ID,
                fields=(("x", IntType()), ("y", IntType())),
                decl_node_id=700000,
            )
        )
        handle = RecordType(name="Point", module_id=ENTRY_ID, decl_id=700000)
        assert dict(table.record_fields(handle)) == {"x": IntType(), "y": IntType()}

    def test_enum_members_non_generic(self) -> None:
        table = TypeTable()
        register_typedef(
            table,
            enum_typedef("Color", {"Red": {}, "Custom": {"hex": TextType()}}, decl_id=700001),
        )
        handle = EnumType(name="Color", module_id=ENTRY_ID, decl_id=700001)
        result = _enum_fields(table, handle)
        assert {v: dict(f) for v, f in result.items()} == {
            "Red": {},
            "Custom": {"hex": TextType()},
        }

    def test_record_fields_missing_def_raises_keyerror(self) -> None:
        table = TypeTable()
        handle = RecordType(name="Ghost", module_id=ENTRY_ID)
        with pytest.raises(KeyError):
            table.record_fields(handle)

    def test_enum_members_missing_def_raises_keyerror(self) -> None:
        table = TypeTable()
        handle = EnumType(name="Ghost", module_id=ENTRY_ID)
        with pytest.raises(KeyError):
            _enum_fields(table, handle)

    def test_record_fields_raises_when_key_registered_as_enum(self) -> None:
        table = TypeTable()
        register_typedef(table, enum_typedef("Color", {"Red": {}}, decl_id=700001))
        handle = RecordType(name="Color", module_id=ENTRY_ID, decl_id=700001)
        with pytest.raises(AssertionError):
            table.record_fields(handle)

    def test_enum_members_raises_when_key_registered_as_record(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Point",
                module_id=ENTRY_ID,
                fields=(("x", IntType()),),
                decl_node_id=700000,
            )
        )
        handle = EnumType(name="Point", module_id=ENTRY_ID, decl_id=700000)
        with pytest.raises(AssertionError):
            _enum_fields(table, handle)


# ---------------------------------------------------------------------------
# Exception accessors — exception_fields (base-chain flattening) and
# exception_def (abstract/base metadata)
# ---------------------------------------------------------------------------


class TestExceptionAccessors:
    def test_exception_fields_root_only(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="Exception",
                module_id=ENTRY_ID,
                fields=(("message", TextType()),),
                abstract=True,
                decl_node_id=700007,
            )
        )
        handle = ExceptionType(name="Exception", module_id=ENTRY_ID, decl_id=700007)
        assert dict(table.exception_fields(handle)) == {"message": TextType()}

    def test_exception_fields_flattens_base_chain_root_mid_leaf_order(self) -> None:
        """A three-level ``extends`` chain flattens base-first, in declaration order."""
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="Root",
                module_id=ENTRY_ID,
                fields=(("message", TextType()),),
                abstract=True,
                decl_node_id=700008,
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Mid",
                module_id=ENTRY_ID,
                fields=(("code", IntType()),),
                base=700008,
                decl_node_id=700009,
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Leaf",
                module_id=ENTRY_ID,
                fields=(("detail", TextType()),),
                base=700009,
                decl_node_id=700010,
            )
        )
        handle = ExceptionType(name="Leaf", module_id=ENTRY_ID, decl_id=700010)
        assert list(table.exception_fields(handle).keys()) == ["message", "code", "detail"]

    def test_exception_fields_resolves_cross_module_base(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="Base",
                module_id=_LIB_ID,
                fields=(("message", TextType()),),
                abstract=True,
                decl_node_id=700011,
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Child",
                module_id=ENTRY_ID,
                fields=(("code", IntType()),),
                base=700011,
                decl_node_id=700012,
            )
        )
        handle = ExceptionType(name="Child", module_id=ENTRY_ID, decl_id=700012)
        assert dict(table.exception_fields(handle)) == {
            "message": TextType(),
            "code": IntType(),
        }

    def test_exception_fields_returns_same_object_for_same_handle(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="Boom",
                module_id=ENTRY_ID,
                fields=(("code", IntType()),),
                decl_node_id=700003,
            )
        )
        handle = ExceptionType(name="Boom", module_id=ENTRY_ID, decl_id=700003)
        first = table.exception_fields(handle)
        second = table.exception_fields(handle)
        assert first is second

    def test_exception_fields_cache_updates_when_a_base_def_is_overwritten(self) -> None:
        """A cached descendant mapping follows a changed base declaration.

        A declaration's shape is immutable once registered (a redeclaration
        always mints a fresh identity -- see ``TypeTable.register``), so
        ``merge_from`` treating another table as authoritative is the one path
        that changes a def under an existing identity. Flattened exception
        caches are keyed by descendant, not by the base that changed, so this
        proves they are dropped wholesale rather than only for the identity
        whose def moved.
        """
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="Base",
                module_id=ENTRY_ID,
                fields=(("old", IntType()),),
                decl_node_id=700013,
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Child",
                module_id=ENTRY_ID,
                fields=(("own", TextType()),),
                base=700013,
                decl_node_id=700012,
            )
        )
        child = ExceptionType(name="Child", module_id=ENTRY_ID, decl_id=700012)
        assert dict(table.exception_fields(child)) == {"old": IntType(), "own": TextType()}

        source = TypeTable()
        source.register(
            TypeDef(
                kind="exception",
                name="Base",
                module_id=ENTRY_ID,
                fields=(("new", BoolType()),),
                decl_node_id=700013,
            )
        )
        table.merge_from(source)

        assert dict(table.exception_fields(child)) == {"new": BoolType(), "own": TextType()}

    def test_exception_field_kinds_cache_updates_when_a_base_def_is_overwritten(self) -> None:
        """The field-kinds memo follows a changed base the same way.

        Same authoritative-``merge_from`` path as the field mapping above,
        against the separate ``exception_field_kinds`` memo.
        """
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="Base",
                module_id=ENTRY_ID,
                fields=(("old", IntType()),),
                field_kinds=("standard",),
                decl_node_id=700013,
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Child",
                module_id=ENTRY_ID,
                fields=(("own", TextType()),),
                field_kinds=("named_only",),
                base=700013,
                decl_node_id=700012,
            )
        )
        child = ExceptionType(name="Child", module_id=ENTRY_ID, decl_id=700012)
        assert table.exception_field_kinds(child) == (("old", "standard"), ("own", "named_only"))

        source = TypeTable()
        source.register(
            TypeDef(
                kind="exception",
                name="Base",
                module_id=ENTRY_ID,
                fields=(("new", BoolType()),),
                field_kinds=("positional_only",),
                decl_node_id=700013,
            )
        )
        table.merge_from(source)

        assert table.exception_field_kinds(child) == (
            ("new", "positional_only"),
            ("own", "named_only"),
        )

    def test_exception_fields_missing_def_raises_keyerror(self) -> None:
        table = TypeTable()
        handle = ExceptionType(name="Ghost", module_id=ENTRY_ID)
        with pytest.raises(KeyError):
            table.exception_fields(handle)

    def test_exception_fields_raises_when_key_registered_as_record(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Point",
                module_id=ENTRY_ID,
                fields=(("x", IntType()),),
                decl_node_id=700000,
            )
        )
        handle = ExceptionType(name="Point", module_id=ENTRY_ID, decl_id=700000)
        with pytest.raises(AssertionError):
            table.exception_fields(handle)

    def test_exception_fields_raises_on_cyclic_base_chain(self) -> None:
        """Internal robustness guard: a cyclic ``base`` chain cannot occur via the
        builder (the temporary recursion ban rejects it first), but the table
        itself still guards against infinite recursion."""
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="A",
                module_id=ENTRY_ID,
                base=700015,
                decl_node_id=700014,
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="B",
                module_id=ENTRY_ID,
                base=700014,
                decl_node_id=700015,
            )
        )
        handle = ExceptionType(name="A", module_id=ENTRY_ID, decl_id=700014)
        with pytest.raises(AssertionError, match="cyclic exception base chain"):
            table.exception_fields(handle)

    def test_exception_def_returns_abstract_and_base(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="Root",
                module_id=ENTRY_ID,
                fields=(("message", TextType()),),
                abstract=True,
                decl_node_id=700008,
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Child",
                module_id=ENTRY_ID,
                fields=(("code", IntType()),),
                base=700008,
                decl_node_id=700012,
            )
        )
        root_def = table.exception_def(
            ExceptionType(name="Root", module_id=ENTRY_ID, decl_id=700008)
        )
        assert root_def.abstract is True
        assert root_def.base is None
        child_def = table.exception_def(
            ExceptionType(name="Child", module_id=ENTRY_ID, decl_id=700012)
        )
        assert child_def.abstract is False
        assert child_def.base == 700008

    def test_exception_def_missing_def_raises_keyerror(self) -> None:
        table = TypeTable()
        handle = ExceptionType(name="Ghost", module_id=ENTRY_ID)
        with pytest.raises(KeyError):
            table.exception_def(handle)

    def test_exception_def_raises_when_key_registered_as_enum(self) -> None:
        table = TypeTable()
        register_typedef(table, enum_typedef("Color", {"Red": {}}, decl_id=700001))
        handle = ExceptionType(name="Color", module_id=ENTRY_ID, decl_id=700001)
        with pytest.raises(AssertionError):
            table.exception_def(handle)


# ---------------------------------------------------------------------------
# Method registry — declarations are plain semantic data keyed by their
# nominal owner; exception owners inherit their base methods.
# ---------------------------------------------------------------------------


class TestMethodRegistry:
    def test_registers_methods_for_record_enum_and_exception_owners(self) -> None:
        table = TypeTable()
        point = RecordType(name="Point", module_id=_LIB_ID, scope_path=("Models",), decl_id=700016)
        color = EnumType(name="Color", module_id=ENTRY_ID, decl_id=700001)
        fault = ExceptionType(name="Fault", module_id=ENTRY_ID, decl_id=700017)
        table.register(
            TypeDef(
                kind="record",
                name="Point",
                module_id=_LIB_ID,
                scope_path=("Models",),
                decl_node_id=700016,
            )
        )
        table.register(TypeDef(kind="enum", name="Color", module_id=ENTRY_ID, decl_node_id=700001))
        table.register(
            TypeDef(kind="exception", name="Fault", module_id=ENTRY_ID, decl_node_id=700017)
        )
        point_shift = MethodDef(
            module_id=_LIB_ID,
            scope_path=("Models", "Point"),
            name="shift",
            decl_node_id=1,
            signature=FunctionType(params=(point, IntType()), result=point),
            receiver_type_param_arity=0,
        )
        color_primary = MethodDef(
            module_id=ENTRY_ID,
            scope_path=("Color",),
            name="primary",
            decl_node_id=2,
            signature=FunctionType(params=(color,), result=BoolType()),
            receiver_type_param_arity=0,
        )
        fault_code = MethodDef(
            module_id=ENTRY_ID,
            scope_path=("Fault",),
            name="code",
            decl_node_id=3,
            signature=FunctionType(params=(fault,), result=IntType()),
            receiver_type_param_arity=0,
        )

        table.register_method(point, point_shift)
        table.register_method(color, color_primary)
        table.register_method(fault, fault_code)
        table.register_method(point, point_shift)

        assert table.lookup_method(point, "shift") == point_shift
        assert table.lookup_method(color, "primary") == color_primary
        assert table.lookup_method(fault, "code") == fault_code
        assert table.methods_for(fault) == table.methods_for(fault) == {"code": fault_code}
        assert point_shift.module_id == _LIB_ID
        assert point_shift.scope_path == ("Models", "Point")
        assert point_shift.name == "shift"
        assert point_shift.signature == FunctionType(params=(point, IntType()), result=point)

    def test_exception_lookup_inherits_a_method_from_a_three_level_base_chain(self) -> None:
        table = TypeTable()
        root = ExceptionType(name="Root", module_id=ENTRY_ID, decl_id=700008)
        leaf = ExceptionType(name="Leaf", module_id=ENTRY_ID, decl_id=700010)
        table.register(
            TypeDef(kind="exception", name="Root", module_id=ENTRY_ID, decl_node_id=700008)
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Mid",
                module_id=ENTRY_ID,
                base=700008,
                decl_node_id=700009,
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Leaf",
                module_id=ENTRY_ID,
                base=700009,
                decl_node_id=700010,
            )
        )
        describe = MethodDef(
            module_id=ENTRY_ID,
            scope_path=("Root",),
            name="describe",
            decl_node_id=1,
            signature=FunctionType(params=(root,), result=TextType()),
            receiver_type_param_arity=0,
        )
        table.register_method(root, describe)

        assert table.lookup_method(leaf, "describe") == describe

    def test_register_method_invalidates_cached_inherited_lookup_miss(self) -> None:
        table = TypeTable()
        base = ExceptionType(name="Base", module_id=ENTRY_ID, decl_id=700013)
        child = ExceptionType(name="Child", module_id=ENTRY_ID, decl_id=700012)
        table.register(
            TypeDef(kind="exception", name="Base", module_id=ENTRY_ID, decl_node_id=700013)
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Child",
                module_id=ENTRY_ID,
                base=700013,
                decl_node_id=700012,
            )
        )

        assert table.lookup_method(child, "status") is None

        status = MethodDef(
            module_id=ENTRY_ID,
            scope_path=("Base",),
            name="status",
            decl_node_id=1,
            signature=FunctionType(params=(base,), result=IntType()),
            receiver_type_param_arity=0,
        )
        table.register_method(base, status)

        assert table.lookup_method(child, "status") == status

    def test_exception_lookup_rejects_a_cyclic_base_chain(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="A",
                module_id=ENTRY_ID,
                base=700015,
                decl_node_id=700014,
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="B",
                module_id=ENTRY_ID,
                base=700014,
                decl_node_id=700015,
            )
        )

        with pytest.raises(AssertionError, match="cyclic exception base chain"):
            table.lookup_method(
                ExceptionType(name="A", module_id=ENTRY_ID, decl_id=700014), "missing"
            )

    def test_lookup_miss_returns_none_for_owner_with_no_methods(self) -> None:
        table = TypeTable()
        point = RecordType(name="Point", module_id=ENTRY_ID, decl_id=700000)
        table.register(
            TypeDef(kind="record", name="Point", module_id=ENTRY_ID, decl_node_id=700000)
        )

        assert table.lookup_method(point, "missing") is None

    def test_merge_from_overwrites_method_entries_and_invalidates_exception_lookup_cache(
        self,
    ) -> None:
        base = ExceptionType(name="Base", module_id=ENTRY_ID, decl_id=700013)
        child = ExceptionType(name="Child", module_id=ENTRY_ID, decl_id=700012)
        target = TypeTable()
        target.register(
            TypeDef(kind="exception", name="Base", module_id=ENTRY_ID, decl_node_id=700013)
        )
        target.register(
            TypeDef(
                kind="exception",
                name="Child",
                module_id=ENTRY_ID,
                base=700013,
                decl_node_id=700012,
            )
        )
        target.register_method(
            base,
            MethodDef(
                module_id=ENTRY_ID,
                scope_path=("Base",),
                name="status",
                decl_node_id=1,
                signature=FunctionType(params=(base,), result=TextType()),
                receiver_type_param_arity=0,
            ),
        )
        table_method = target.lookup_method(child, "status")
        assert table_method is not None
        assert table_method.signature.result == TextType()

        source = TypeTable()
        source.register(
            TypeDef(kind="exception", name="Base", module_id=ENTRY_ID, decl_node_id=700013)
        )
        replacement = MethodDef(
            module_id=ENTRY_ID,
            scope_path=("Base",),
            name="status",
            decl_node_id=2,
            signature=FunctionType(params=(base,), result=IntType()),
            receiver_type_param_arity=0,
        )
        source.register_method(base, replacement)
        target.merge_from(source)

        assert target.lookup_method(child, "status") == replacement

    def test_merge_from_skips_identical_method_entries(self) -> None:
        fault = ExceptionType(name="Fault", module_id=ENTRY_ID, decl_id=700017)
        method = MethodDef(
            module_id=ENTRY_ID,
            scope_path=("Fault",),
            name="status",
            decl_node_id=1,
            signature=FunctionType(params=(fault,), result=IntType()),
            receiver_type_param_arity=0,
        )
        source = TypeTable()
        source.register(
            TypeDef(kind="exception", name="Fault", module_id=ENTRY_ID, decl_node_id=700017)
        )
        source.register_method(fault, method)
        target = TypeTable()
        target.register(
            TypeDef(kind="exception", name="Fault", module_id=ENTRY_ID, decl_node_id=700017)
        )
        target.register_method(fault, method)
        before = dict(target.methods_for(fault))

        target.merge_from(source)

        assert dict(target.methods_for(fault)) == before == {"status": method}
        assert target.lookup_method(fault, "status") == method

    def test_merge_from_drops_inherited_methods_of_an_overwritten_base_def(self) -> None:
        """An overwritten base def takes its own methods, and the memo, with it.

        The authoritative table carries no method for that identity, so
        nothing re-registers one: only dropping the base's direct map AND the
        flattened descendant memo can stop the inherited entry from still
        answering.
        """
        base = ExceptionType(name="Base", module_id=ENTRY_ID, decl_id=700013)
        child = ExceptionType(name="Child", module_id=ENTRY_ID, decl_id=700012)
        target = TypeTable()
        target.register(
            TypeDef(kind="exception", name="Base", module_id=ENTRY_ID, decl_node_id=700013)
        )
        target.register(
            TypeDef(
                kind="exception",
                name="Child",
                module_id=ENTRY_ID,
                base=700013,
                decl_node_id=700012,
            )
        )
        target.register_method(
            base,
            MethodDef(
                module_id=ENTRY_ID,
                scope_path=("Base",),
                name="message",
                decl_node_id=1,
                signature=FunctionType(params=(base,), result=TextType()),
                receiver_type_param_arity=0,
            ),
        )
        assert target.lookup_method(child, "message") is not None

        source = TypeTable()
        source.register(
            TypeDef(
                kind="exception",
                name="Base",
                module_id=ENTRY_ID,
                fields=(("code", IntType()),),
                decl_node_id=700013,
            )
        )
        target.merge_from(source)

        assert target.lookup_method(child, "message") is None

    def test_generic_owner_records_receiver_type_parameter_arity(self) -> None:
        table = TypeTable()
        box = RecordType(
            name="Box", type_args=(TypeVarType("T"),), module_id=ENTRY_ID, decl_id=700002
        )
        table.register(
            TypeDef(
                kind="record",
                name="Box",
                module_id=ENTRY_ID,
                type_params=("T",),
                decl_node_id=700002,
            )
        )
        get = MethodDef(
            module_id=ENTRY_ID,
            scope_path=("Box",),
            name="get",
            decl_node_id=1,
            signature=FunctionType(params=(box,), result=TypeVarType("T")),
            receiver_type_param_arity=1,
        )
        table.register_method(box, get)

        found = table.lookup_method(
            RecordType(name="Box", module_id=ENTRY_ID, decl_id=700002), "get"
        )
        assert found == get
        assert found.receiver_type_param_arity == 1


# ---------------------------------------------------------------------------
# TypeTable.exception_field_kinds — own fields honor their declared kind;
# only the extends-chain flattening order (base-first) is exception-specific.
# ---------------------------------------------------------------------------


class TestExceptionFieldKinds:
    """``exception_field_kinds`` returns ``ParamKind.value`` strings (not the
    enum): ``semantics`` may not import ``syntax.nodes``, so ``TypeDef.
    field_kinds`` stores the stable string values instead (converted back to
    ``ParamKind`` by ``typecheck.env``)."""

    def test_root_only_returns_declared_field_kinds(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="Exception",
                module_id=ENTRY_ID,
                fields=(("message", TextType()),),
                abstract=True,
                field_kinds=(ParamKind.NAMED_ONLY.value,),
                decl_node_id=700007,
            )
        )
        handle = ExceptionType(name="Exception", module_id=ENTRY_ID, decl_id=700007)
        assert table.exception_field_kinds(handle) == (("message", ParamKind.NAMED_ONLY.value),)

    def test_flattens_base_chain_and_honors_each_level_own_marker(self) -> None:
        """Own fields honor their declared kind at every level of the chain —
        an exception's own fields are not forced to NAMED_ONLY, only the
        flattening order (base-first) is exception-specific."""
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="Root",
                module_id=ENTRY_ID,
                fields=(("message", TextType()),),
                abstract=True,
                field_kinds=(ParamKind.NAMED_ONLY.value,),
                decl_node_id=700008,
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Mid",
                module_id=ENTRY_ID,
                fields=(("code", IntType()),),
                base=700008,
                field_kinds=(ParamKind.STANDARD.value,),
                decl_node_id=700009,
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Leaf",
                module_id=ENTRY_ID,
                fields=(("detail", TextType()),),
                base=700009,
                field_kinds=(ParamKind.POSITIONAL_ONLY.value,),
                decl_node_id=700010,
            )
        )
        handle = ExceptionType(name="Leaf", module_id=ENTRY_ID, decl_id=700010)
        assert table.exception_field_kinds(handle) == (
            ("message", ParamKind.NAMED_ONLY.value),
            ("code", ParamKind.STANDARD.value),
            ("detail", ParamKind.POSITIONAL_ONLY.value),
        )

    def test_resolves_cross_module_base(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="Base",
                module_id=_LIB_ID,
                fields=(("message", TextType()),),
                abstract=True,
                field_kinds=(ParamKind.NAMED_ONLY.value,),
                decl_node_id=700011,
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Child",
                module_id=ENTRY_ID,
                fields=(("code", IntType()),),
                base=700011,
                field_kinds=(ParamKind.STANDARD.value,),
                decl_node_id=700012,
            )
        )
        handle = ExceptionType(name="Child", module_id=ENTRY_ID, decl_id=700012)
        assert table.exception_field_kinds(handle) == (
            ("message", ParamKind.NAMED_ONLY.value),
            ("code", ParamKind.STANDARD.value),
        )

    def test_returns_same_object_for_same_handle(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="Boom",
                module_id=ENTRY_ID,
                fields=(("code", IntType()),),
                field_kinds=(ParamKind.STANDARD.value,),
                decl_node_id=700003,
            )
        )
        handle = ExceptionType(name="Boom", module_id=ENTRY_ID, decl_id=700003)
        first = table.exception_field_kinds(handle)
        second = table.exception_field_kinds(handle)
        assert first is second

    def test_missing_def_raises_keyerror(self) -> None:
        table = TypeTable()
        handle = ExceptionType(name="Ghost", module_id=ENTRY_ID)
        with pytest.raises(KeyError):
            table.exception_field_kinds(handle)

    def test_raises_when_key_registered_as_record(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Point",
                module_id=ENTRY_ID,
                fields=(("x", IntType()),),
                decl_node_id=700000,
            )
        )
        handle = ExceptionType(name="Point", module_id=ENTRY_ID, decl_id=700000)
        with pytest.raises(AssertionError):
            table.exception_field_kinds(handle)

    def test_raises_on_cyclic_base_chain(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="A",
                module_id=ENTRY_ID,
                base=700015,
                decl_node_id=700014,
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="B",
                module_id=ENTRY_ID,
                base=700014,
                decl_node_id=700015,
            )
        )
        handle = ExceptionType(name="A", module_id=ENTRY_ID, decl_id=700014)
        with pytest.raises(AssertionError, match="cyclic exception base chain"):
            table.exception_field_kinds(handle)


# ---------------------------------------------------------------------------
# Substitution on generic handles
# ---------------------------------------------------------------------------


class TestGenericSubstitution:
    def test_record_fields_substitutes_nested_containers(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Pair",
                module_id=ENTRY_ID,
                type_params=("T", "U"),
                fields=(
                    ("first", TypeVarType("T")),
                    ("second", TypeVarType("U")),
                    ("firsts", ArrayType(TypeVarType("T"))),
                    ("seconds", DictType(TypeVarType("U"))),
                ),
                decl_node_id=700018,
            )
        )
        handle = RecordType(
            name="Pair", type_args=(IntType(), TextType()), module_id=ENTRY_ID, decl_id=700018
        )
        result = table.record_fields(handle)
        assert dict(result) == {
            "first": IntType(),
            "second": TextType(),
            "firsts": ArrayType(IntType()),
            "seconds": DictType(TextType()),
        }

    def test_enum_members_substitute_type_args(self) -> None:
        table = TypeTable()
        register_typedef(
            table,
            enum_typedef(
                "Maybe",
                {"None": {}, "Just": {"value": TypeVarType("T")}},
                type_params=("T",),
                decl_id=700019,
            ),
        )
        handle = EnumType(name="Maybe", type_args=(IntType(),), module_id=ENTRY_ID, decl_id=700019)
        result = _enum_fields(table, handle)
        assert {v: dict(f) for v, f in result.items()} == {
            "None": {},
            "Just": {"value": IntType()},
        }


# ---------------------------------------------------------------------------
# Record field mutability
# ---------------------------------------------------------------------------


class TestRecordMutableFields:
    def test_accessor_preserves_declaration_names_across_generic_instantiations(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Pair",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(("mutable", TypeVarType("T")), ("fixed", IntType())),
                mutable_fields=frozenset({"mutable"}),
                decl_node_id=700020,
            )
        )
        handle = RecordType(
            name="Pair", type_args=(TextType(),), module_id=ENTRY_ID, decl_id=700020
        )

        assert table.record_mutable_fields(handle) == frozenset({"mutable"})
        assert table.record_mutable_fields(
            RecordType(name="Pair", type_args=(IntType(),), module_id=ENTRY_ID, decl_id=700020)
        ) == frozenset({"mutable"})
        assert dict(table.record_fields(handle)) == {"mutable": TextType(), "fixed": IntType()}

    def test_accessor_rejects_missing_and_non_record_definitions(self) -> None:
        table = TypeTable()
        with pytest.raises(KeyError):
            table.record_mutable_fields(RecordType(name="Missing", module_id=ENTRY_ID))

        table.register(TypeDef(kind="enum", name="Kind", module_id=ENTRY_ID, decl_node_id=700023))
        with pytest.raises(AssertionError):
            table.record_mutable_fields(RecordType(name="Kind", module_id=ENTRY_ID, decl_id=700023))

    def test_builder_registers_standalone_and_enum_member_mutability(self) -> None:
        checked = _check(
            "record Standalone(var value: int, label: text)\n"
            "record Referenced(var value: int)\n"
            "enum Members\n"
            "  | Inline(var value: int)\n"
            "  | ::Referenced\n"
            "enum Generic[T] | Box(var value: T)\n"
            "()"
        )
        table = checked.type_env.type_table
        standalone = checked.type_env.get_type("Standalone")
        referenced = checked.type_env.get_type("Referenced")
        generic = checked.type_env.get_generic_type("Generic")

        assert isinstance(standalone, RecordType)
        assert isinstance(referenced, RecordType)
        assert generic is not None
        assert table.record_mutable_fields(standalone) == frozenset({"value"})
        assert table.record_mutable_fields(referenced) == frozenset({"value"})
        box = table.enum_member_names(generic.template)["Box"]
        assert table.record_mutable_fields(box) == frozenset({"value"})

    def test_seeded_builtin_records_are_immutable(self) -> None:
        table = create_seeded_type_table()
        handle = BUILTIN_PRELUDE_TYPES["ExecResult"]

        assert isinstance(handle, RecordType)
        assert table.record_mutable_fields(handle) == frozenset()

    def test_merge_replaces_record_mutable_fields(self) -> None:
        target = TypeTable()
        handle = RecordType(name="Point", module_id=ENTRY_ID, decl_id=700021)
        target.register(
            TypeDef(
                kind="record",
                name="Point",
                module_id=ENTRY_ID,
                fields=(("value", IntType()),),
                decl_node_id=700021,
            )
        )
        assert target.record_mutable_fields(handle) == frozenset()

        source = TypeTable()
        source.register(
            TypeDef(
                kind="record",
                name="Point",
                module_id=ENTRY_ID,
                fields=(("value", IntType()),),
                mutable_fields=frozenset({"value"}),
                decl_node_id=700021,
            )
        )
        target.merge_from(source)

        assert target.record_mutable_fields(handle) == frozenset({"value"})

    def test_mutability_participates_in_builtin_shape_validation(self) -> None:
        with pytest.raises(AglTypeError):
            _check(
                "builtin record ExecResult\n"
                "  var stdout: text\n"
                "  exit_code: int\n"
                "  stderr: text\n"
                "  timed_out: bool\n"
                "()",
                default_stdlib=False,
            )

    def test_mutability_participates_in_typedef_shape(self) -> None:
        fixed = TypeDef(
            kind="record",
            name="Point",
            module_id=ENTRY_ID,
            fields=(("value", IntType()),),
            decl_node_id=700022,
        )
        mutable = replace(fixed, mutable_fields=frozenset({"value"}))

        assert fixed != mutable


# ---------------------------------------------------------------------------
# Memoization
# ---------------------------------------------------------------------------


class TestMemoization:
    def test_record_fields_returns_same_object_for_same_handle(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Point",
                module_id=ENTRY_ID,
                fields=(("x", IntType()),),
                decl_node_id=700000,
            )
        )
        handle = RecordType(name="Point", module_id=ENTRY_ID, decl_id=700000)
        first = table.record_fields(handle)
        second = table.record_fields(handle)
        assert first is second

    def test_enum_members_return_same_object_for_same_handle(self) -> None:
        table = TypeTable()
        register_typedef(table, enum_typedef("Color", {"Red": {}}, decl_id=700001))
        handle = EnumType(name="Color", module_id=ENTRY_ID, decl_id=700001)
        first = table.enum_members(handle)
        second = table.enum_members(handle)
        assert first is second

    def test_record_fields_caches_each_generic_instantiation_separately(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Box",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(("value", TypeVarType("T")),),
                decl_node_id=700002,
            )
        )
        int_handle = RecordType(
            name="Box", type_args=(IntType(),), module_id=ENTRY_ID, decl_id=700002
        )
        text_handle = RecordType(
            name="Box", type_args=(TextType(),), module_id=ENTRY_ID, decl_id=700002
        )
        assert dict(table.record_fields(int_handle)) == {"value": IntType()}
        assert dict(table.record_fields(text_handle)) == {"value": TextType()}
        # Re-fetching the first handle still returns its own cached result.
        assert dict(table.record_fields(int_handle)) == {"value": IntType()}

    def test_enum_members_cache_each_generic_instantiation_separately(self) -> None:
        table = TypeTable()
        register_typedef(
            table,
            enum_typedef(
                "Maybe",
                {"Just": {"value": TypeVarType("T")}},
                type_params=("T",),
                decl_id=700019,
            ),
        )
        int_handle = EnumType(
            name="Maybe", type_args=(IntType(),), module_id=ENTRY_ID, decl_id=700019
        )
        text_handle = EnumType(
            name="Maybe", type_args=(TextType(),), module_id=ENTRY_ID, decl_id=700019
        )
        assert {v: dict(f) for v, f in _enum_fields(table, int_handle).items()} == {
            "Just": {"value": IntType()}
        }
        assert {v: dict(f) for v, f in _enum_fields(table, text_handle).items()} == {
            "Just": {"value": TextType()}
        }


# ---------------------------------------------------------------------------
# Re-registration semantics
# ---------------------------------------------------------------------------


class TestReRegistration:
    def test_identical_re_registration_is_a_no_op(self) -> None:
        table = TypeTable()
        typedef = TypeDef(
            kind="record",
            name="Point",
            module_id=ENTRY_ID,
            fields=(("x", IntType()),),
            decl_node_id=700000,
        )
        table.register(typedef)
        table.register(
            TypeDef(
                kind="record",
                name="Point",
                module_id=ENTRY_ID,
                fields=(("x", IntType()),),
                decl_node_id=700000,
            )
        )
        assert table.get(ENTRY_ID, "Point") == typedef

    def test_conflicting_re_registration_raises(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Point",
                module_id=ENTRY_ID,
                fields=(("x", IntType()),),
                decl_node_id=700000,
            )
        )
        with pytest.raises(AssertionError):
            table.register(
                TypeDef(
                    kind="record",
                    name="Point",
                    module_id=ENTRY_ID,
                    fields=(("x", TextType()),),
                    decl_node_id=700000,
                )
            )

    def test_register_rejects_a_def_with_no_declaration_identity(self) -> None:
        table = TypeTable()
        with pytest.raises(AssertionError, match="no declaration identity"):
            table.register(
                TypeDef(kind="record", name="X", module_id=ENTRY_ID, fields=(("a", IntType()),))
            )


# ---------------------------------------------------------------------------
# Declaration identity: two declarations can share one name path
# ---------------------------------------------------------------------------


class TestSupersession:
    """A name is a pointer to the newest declaration bearing it; an older
    declaration is superseded, never removed — it stays registered, own
    identity intact, for any handle that still names it."""

    def test_two_declarations_sharing_a_name_path_coexist_in_one_table(self) -> None:
        table = TypeTable()
        old_def = TypeDef(
            kind="record",
            name="Point",
            module_id=ENTRY_ID,
            fields=(("x", IntType()),),
            decl_node_id=700300,
        )
        new_def = TypeDef(
            kind="record",
            name="Point",
            module_id=ENTRY_ID,
            fields=(("y", TextType()),),
            decl_node_id=700301,
        )
        table.register(old_def)
        table.register(new_def)

        # Both declarations remain retrievable by their own identity, with
        # their own field shapes.
        assert table.get_by_id(700300) == old_def
        assert table.get_by_id(700301) == new_def
        old_handle = RecordType(name="Point", module_id=ENTRY_ID, decl_id=700300)
        new_handle = RecordType(name="Point", module_id=ENTRY_ID, decl_id=700301)
        assert dict(table.record_fields(old_handle)) == {"x": IntType()}
        assert dict(table.record_fields(new_handle)) == {"y": TextType()}

        # The two handles name distinct declarations, so they compare unequal
        # even though every other field matches.
        assert old_handle != new_handle

        # A name lookup answers with the newest declaration.
        assert table.get(ENTRY_ID, "Point") == new_def

    def test_superseded_declaration_keeps_its_own_methods_and_exception_base_chain(self) -> None:
        table = TypeTable()
        root_one = TypeDef(
            kind="exception", name="RootOne", module_id=ENTRY_ID, abstract=True, decl_node_id=700302
        )
        root_two = TypeDef(
            kind="exception", name="RootTwo", module_id=ENTRY_ID, abstract=True, decl_node_id=700303
        )
        table.register(root_one)
        table.register(root_two)

        old_fault = TypeDef(
            kind="exception",
            name="Fault",
            module_id=ENTRY_ID,
            base=700302,
            decl_node_id=700304,
        )
        new_fault = TypeDef(
            kind="exception",
            name="Fault",
            module_id=ENTRY_ID,
            base=700303,
            decl_node_id=700305,
        )
        table.register(old_fault)
        table.register(new_fault)

        old_handle = ExceptionType(name="Fault", module_id=ENTRY_ID, decl_id=700304)
        new_handle = ExceptionType(name="Fault", module_id=ENTRY_ID, decl_id=700305)
        describe = MethodDef(
            module_id=ENTRY_ID,
            scope_path=("Fault",),
            name="describe",
            decl_node_id=1,
            signature=FunctionType(params=(old_handle,), result=TextType()),
            receiver_type_param_arity=0,
        )
        explain = MethodDef(
            module_id=ENTRY_ID,
            scope_path=("Fault",),
            name="explain",
            decl_node_id=2,
            signature=FunctionType(params=(new_handle,), result=TextType()),
            receiver_type_param_arity=0,
        )
        table.register_method(old_handle, describe)
        table.register_method(new_handle, explain)

        # Each declaration owns only its own method...
        assert table.lookup_method(old_handle, "describe") == describe
        assert table.lookup_method(old_handle, "explain") is None
        assert table.lookup_method(new_handle, "explain") == explain
        assert table.lookup_method(new_handle, "describe") is None

        # ...and its own exception base chain.
        assert table.exception_def(old_handle).base == 700302
        assert table.exception_def(new_handle).base == 700303

    def test_exception_whose_base_is_superseded_keeps_inheriting_from_its_own_base(self) -> None:
        table = TypeTable()
        old_root = TypeDef(
            kind="exception",
            name="Root",
            module_id=ENTRY_ID,
            fields=(("code", IntType()),),
            abstract=True,
            decl_node_id=700306,
        )
        table.register(old_root)
        child = TypeDef(
            kind="exception",
            name="Child",
            module_id=ENTRY_ID,
            base=700306,
            decl_node_id=700307,
        )
        table.register(child)

        # Redeclaring "Root" under a fresh identity supersedes the name, but
        # Child's own base link still names the declaration it was built
        # against.
        new_root = TypeDef(
            kind="exception",
            name="Root",
            module_id=ENTRY_ID,
            fields=(("message", TextType()),),
            abstract=True,
            decl_node_id=700308,
        )
        table.register(new_root)

        assert table.get(ENTRY_ID, "Root") == new_root
        child_handle = ExceptionType(name="Child", module_id=ENTRY_ID, decl_id=700307)
        assert dict(table.exception_fields(child_handle)) == {"code": IntType()}

    def test_whole_table_fixpoint_answers_each_identity_from_its_own_body(self) -> None:
        table = TypeTable()
        old_def = TypeDef(
            kind="record",
            name="Box",
            module_id=ENTRY_ID,
            fields=(("slot", IntType()),),
            decl_node_id=700310,
        )
        table.register(old_def)
        old_handle = RecordType(name="Box", module_id=ENTRY_ID, decl_id=700310)
        # Populate the cached non-data fixpoint before the redeclaration.
        assert not table.nominal_reaches_non_data(old_handle)

        new_def = TypeDef(
            kind="record",
            name="Box",
            module_id=ENTRY_ID,
            fields=(("slot", UnitType()),),
            decl_node_id=700311,
        )
        table.register(new_def)
        new_handle = RecordType(name="Box", module_id=ENTRY_ID, decl_id=700311)

        # The registration invalidated the cached fixpoint, which now covers
        # both declarations: each identity answers from its OWN body, rather
        # than both collapsing onto whichever one the shared name resolves to.
        assert table.nominal_reaches_non_data(new_handle)
        assert not table.nominal_reaches_non_data(old_handle)


# ---------------------------------------------------------------------------
# entries() / merge_from()
# ---------------------------------------------------------------------------


class TestEntriesAndMerge:
    def test_entries_returns_all_registered_defs(self) -> None:
        table = TypeTable()
        a = TypeDef(kind="record", name="A", module_id=ENTRY_ID, fields=(), decl_node_id=700014)
        b = TypeDef(kind="record", name="B", module_id=ENTRY_ID, fields=(), decl_node_id=700015)
        table.register(a)
        table.register(b)
        assert set(table.entries()) == {a, b}

    def test_merge_from_copies_new_entries_and_skips_identical(self) -> None:
        source = TypeTable()
        shared = TypeDef(
            kind="record", name="Shared", module_id=ENTRY_ID, fields=(), decl_node_id=700020
        )
        new = TypeDef(kind="record", name="New", module_id=ENTRY_ID, fields=(), decl_node_id=700021)
        source.register(shared)
        source.register(new)

        target = TypeTable()
        target.register(shared)  # already present, identical

        target.merge_from(source)

        assert target.get(ENTRY_ID, "Shared") == shared
        assert target.get(ENTRY_ID, "New") == new

    def test_merge_from_overwrites_conflicting_entry_with_others_value(self) -> None:
        # merge_from treats the source as authoritative: unlike a direct
        # register() call, a conflicting entry does not raise — it is
        # overwritten.  This is required for REPL redefinition (see
        # TestReplSeeding.test_redefined_record_updates_session_table): the
        # persistent session table must adopt a later entry's redefinition of
        # a previously-declared name.
        source = TypeTable()
        new_def = TypeDef(
            kind="record",
            name="X",
            module_id=ENTRY_ID,
            fields=(("a", IntType()),),
            decl_node_id=700022,
        )
        source.register(new_def)

        target = TypeTable()
        target.register(
            TypeDef(
                kind="record",
                name="X",
                module_id=ENTRY_ID,
                fields=(("a", TextType()),),
                decl_node_id=700022,
            )
        )

        target.merge_from(source)

        assert target.get(ENTRY_ID, "X") == new_def

    def test_merge_from_repoints_the_name_index_and_keeps_the_displaced_def(self) -> None:
        # "Other wins" covers the name index as well as the def map: a source
        # declaration under a name path this table already binds to a
        # DIFFERENT identity takes the name over, while the identity it
        # displaces stays registered under its own shape.
        source = TypeTable()
        incoming = TypeDef(
            kind="record",
            name="Point",
            module_id=ENTRY_ID,
            fields=(("y", TextType()),),
            decl_node_id=700402,
        )
        source.register(incoming)

        target = TypeTable()
        displaced = TypeDef(
            kind="record",
            name="Point",
            module_id=ENTRY_ID,
            fields=(("x", IntType()),),
            decl_node_id=700403,
        )
        target.register(displaced)

        target.merge_from(source)

        assert target.get(ENTRY_ID, "Point") == incoming
        assert target.get_by_id(700403) == displaced
        displaced_handle = RecordType(name="Point", module_id=ENTRY_ID, decl_id=700403)
        assert dict(target.record_fields(displaced_handle)) == {"x": IntType()}

    def test_merge_from_keeps_cached_substitution_when_entry_is_unchanged(self) -> None:
        # An identical incoming entry must not perturb an already-cached
        # substitution: the same handle keeps resolving to the same object.
        shared = TypeDef(
            kind="record",
            name="Shared",
            module_id=ENTRY_ID,
            fields=(("a", IntType()),),
            decl_node_id=700020,
        )
        source = TypeTable()
        source.register(shared)

        target = TypeTable()
        target.register(shared)
        handle = RecordType(name="Shared", module_id=ENTRY_ID, decl_id=700020)
        before = target.record_fields(handle)

        target.merge_from(source)

        after = target.record_fields(handle)
        assert after is before

    def test_merge_from_invalidates_stale_cached_substitution(self) -> None:
        source = TypeTable()
        new_def = TypeDef(
            kind="record",
            name="X",
            module_id=ENTRY_ID,
            fields=(("a", IntType()),),
            decl_node_id=700022,
        )
        source.register(new_def)

        target = TypeTable()
        target.register(
            TypeDef(
                kind="record",
                name="X",
                module_id=ENTRY_ID,
                fields=(("a", TextType()),),
                decl_node_id=700022,
            )
        )
        handle = RecordType(name="X", module_id=ENTRY_ID, decl_id=700022)
        assert dict(target.record_fields(handle)) == {"a": TextType()}

        target.merge_from(source)

        assert dict(target.record_fields(handle)) == {"a": IntType()}

    def test_merge_from_invalidates_stale_cached_enum_substitution(self) -> None:
        # The enum memo is bucketed by identity separately from the record
        # one, so an authoritative overwrite has to drop it on its own.
        source = TypeTable()
        register_typedef(source, enum_typedef("Color", {"Red": {}, "Blue": {}}, decl_id=700023))

        target = TypeTable()
        register_typedef(target, enum_typedef("Color", {"Red": {}}, decl_id=700023))
        handle = EnumType(name="Color", module_id=ENTRY_ID, decl_id=700023)
        assert set(target.enum_member_names(handle)) == {"Red"}

        target.merge_from(source)

        assert set(target.enum_member_names(handle)) == {"Red", "Blue"}


# ---------------------------------------------------------------------------
# Built-in prelude seeding
# ---------------------------------------------------------------------------


class TestBuiltinSeeding:
    def test_all_prelude_types_resolvable(self) -> None:
        table = create_seeded_type_table()
        for name, typ in BUILTIN_PRELUDE_TYPES.items():
            typedef = table.get(RESERVED_ID, name)
            assert typedef is not None
            expected = BUILTIN_PRELUDE_TYPE_DEFS[name]
            if isinstance(typ, RecordType):
                handle = RecordType(name=name, module_id=RESERVED_ID, decl_id=typedef.decl_node_id)
                assert dict(table.record_fields(handle)) == dict(expected.fields)
            elif isinstance(typ, ExceptionType):
                handle = ExceptionType(
                    name=name, module_id=RESERVED_ID, decl_id=typedef.decl_node_id
                )
                assert table.exception_def(handle) == expected
            else:
                handle = EnumType(name=name, module_id=RESERVED_ID, decl_id=typedef.decl_node_id)
                result = _enum_fields(table, handle)
                assert {v: dict(f) for v, f in result.items()} == {
                    member.name: dict(table.record_fields(member)) for member in expected.members
                }

    def test_generic_option_seeded_under_the_reserved_sentinel(self) -> None:
        table = create_seeded_type_table()
        typedef = table.get(RESERVED_ID, "Option")
        assert typedef is not None
        assert typedef.type_params == ("T",)
        handle = EnumType(
            name="Option",
            type_args=(TextType(),),
            module_id=RESERVED_ID,
            decl_id=typedef.decl_node_id,
        )
        result = _enum_fields(table, handle)
        assert {v: dict(f) for v, f in result.items()} == {
            "None": {},
            "Some": {"value": TextType()},
        }


# ---------------------------------------------------------------------------
# Every env-registered type has a matching table def: single-module
# ---------------------------------------------------------------------------


class TestEnvTypeHasMatchingTableDefSingleModule:
    """Every type registered in the env has a matching ``TypeDef`` in the
    shared table (the only place shapes live, since handles carry none)."""

    def test_non_generic_record(self) -> None:
        checked = _check("record Point\n  x: int\n  y: int\nlet p = Point(x = 1, y = 2)\np")
        point = checked.type_env.get_type("Point")
        assert isinstance(point, RecordType)
        table = checked.type_env.type_table
        typedef = table.get(point.module_id, "Point")
        assert typedef is not None
        assert typedef.kind == "record"
        assert dict(table.record_fields(point)) == {"x": IntType(), "y": IntType()}

    def test_generic_record(self) -> None:
        checked = _check("record Box[T]\n  value: T\nlet b: Box[int] = Box(value = 1)\nb")
        box = _binding_value_type(checked, "b")
        assert isinstance(box, RecordType)
        table = checked.type_env.type_table
        typedef = table.get(box.module_id, "Box")
        assert typedef is not None
        assert typedef.type_params == ("T",)
        assert dict(table.record_fields(box)) == {"value": IntType()}

    def test_non_generic_enum(self) -> None:
        checked = _check("enum Color\n  | Red\n  | Green\n  | Blue\nlet c = Red\nc")
        color = checked.type_env.get_type("Color")
        assert isinstance(color, EnumType)
        table = checked.type_env.type_table
        typedef = table.get(color.module_id, "Color")
        assert typedef is not None
        assert typedef.kind == "enum"
        result = _enum_fields(table, color)
        assert {v: dict(f) for v, f in result.items()} == {
            "Red": {},
            "Green": {},
            "Blue": {},
        }

    def test_generic_enum(self) -> None:
        checked = _check("enum Maybe[T]\n  | none\n  | just(value: T)\nlet m = just(value = 1)\nm")
        member = _binding_value_type(checked, "m")
        assert isinstance(member, RecordType)
        table = checked.type_env.type_table
        typedef = table.get(ENTRY_ID, "Maybe")
        assert typedef is not None
        assert typedef.type_params == ("T",)
        maybe = checked.type_env.instantiate_nominal("Maybe", (IntType(),))
        assert isinstance(maybe, EnumType)
        result = _enum_fields(table, maybe)
        assert {v: dict(f) for v, f in result.items()} == {
            "none": {},
            "just": {"value": IntType()},
        }


# ---------------------------------------------------------------------------
# Every env-registered type has a matching table def: program context
# ---------------------------------------------------------------------------


class TestEnvTypeHasMatchingTableDefGraphMode:
    def test_record_declared_in_one_module_reachable_from_another(self, tmp_path: Path) -> None:
        modules = {
            "entry": (
                "import mylib\ndef make() -> mylib::Point = mylib::makePoint()\nlet p = make()\np"
            ),
            "mylib": (
                "record Point\n  x: int\n  y: int\ndef makePoint() -> Point = Point(x = 1, y = 2)"
            ),
        }
        cg = _check_program(tmp_path, modules)
        mylib_id = ModuleId.from_path("mylib")
        point = cg.program_type_table[(mylib_id, "Point")]
        assert isinstance(point, RecordType)

        mylib_table = cg.modules[mylib_id].type_env.type_table
        typedef = mylib_table.get(mylib_id, "Point")
        assert typedef is not None
        assert dict(mylib_table.record_fields(point)) == {"x": IntType(), "y": IntType()}

        # The table is shared graph-wide: the entry module's env reaches the
        # same def for mylib's type.
        entry_table = cg.modules[ENTRY_ID].type_env.type_table
        assert entry_table.get(mylib_id, "Point") == mylib_table.get(mylib_id, "Point")


# ---------------------------------------------------------------------------
# REPL accumulation: seed_from carries the table across entries
# ---------------------------------------------------------------------------


class TestReplSeeding:
    def test_session_table_keeps_def_after_later_entry(self) -> None:
        s = ReplSession()
        declare = s.eval_entry("record R\n  a: int")
        assert declare.ok
        use = s.eval_entry("let r = R(a = 1)")
        assert use.ok

        typedef = s._type_env.type_table.get(ENTRY_ID, "R")
        assert typedef is not None
        assert dict(typedef.fields) == {"a": IntType()}

    def test_redefined_record_updates_session_table(self) -> None:
        # Regression: redeclaring a record with a different shape in a later
        # entry (a supported REPL workflow — see TestRedefinition in
        # test_agl_repl_session.py) must update the persisted def rather than
        # raising or leaving the stale shape behind.
        s = ReplSession()
        first = s.eval_entry("record R\n  a: int")
        assert first.ok
        second = s.eval_entry("record R\n  b: text")
        assert second.ok

        typedef = s._type_env.type_table.get(ENTRY_ID, "R")
        assert typedef is not None
        assert dict(typedef.fields) == {"b": TextType()}


# ---------------------------------------------------------------------------
# comparable_types / _reaches_non_data: table-aware record/enum walk
# ---------------------------------------------------------------------------


class TestComparableTypesTableAware:
    def test_record_with_only_scalar_fields_comparable(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Point",
                module_id=ENTRY_ID,
                fields=(("x", IntType()), ("y", IntType())),
                decl_node_id=700000,
            )
        )
        handle = RecordType(name="Point", module_id=ENTRY_ID, decl_id=700000)
        assert comparable_types(handle, handle, table) is True

    def test_generic_enum_function_variant_via_instantiation_not_comparable(self) -> None:
        table = TypeTable()
        register_typedef(
            table,
            enum_typedef(
                "Holder",
                {"None": {}, "Some": {"value": TypeVarType("T")}},
                type_params=("T",),
                decl_id=700023,
            ),
        )
        fn_type = FunctionType(params=(IntType(),), result=IntType())
        fn_handle = EnumType(
            name="Holder", type_args=(fn_type,), module_id=ENTRY_ID, decl_id=700023
        )
        assert comparable_types(fn_handle, fn_handle, table) is False

        text_handle = EnumType(
            name="Holder", type_args=(TextType(),), module_id=ENTRY_ID, decl_id=700023
        )
        assert comparable_types(text_handle, text_handle, table) is True

    def test_record_with_unit_nested_in_array_field_not_comparable(self) -> None:
        # Nested depth: the record field itself is an array, whose element type
        # is the type parameter — instantiating with unit makes the array of
        # unit values transitively non-comparable.
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Wrapper",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(("items", ArrayType(TypeVarType("T"))),),
                decl_node_id=700024,
            )
        )
        handle = RecordType(
            name="Wrapper", type_args=(UnitType(),), module_id=ENTRY_ID, decl_id=700024
        )
        assert comparable_types(handle, handle, table) is False

    def test_exception_with_function_field_not_comparable(self) -> None:
        table = TypeTable()
        handler_type = FunctionType(params=(), result=IntType())
        table.register(
            TypeDef(
                kind="exception",
                name="Failure",
                module_id=ENTRY_ID,
                fields=(("handler", handler_type),),
                decl_node_id=700025,
            )
        )
        exc = ExceptionType(name="Failure", module_id=ENTRY_ID, decl_id=700025)
        assert comparable_types(exc, exc, table) is False

    def test_exception_with_only_scalar_fields_comparable(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="Failure",
                module_id=ENTRY_ID,
                fields=(("code", IntType()),),
                decl_node_id=700025,
            )
        )
        exc = ExceptionType(name="Failure", module_id=ENTRY_ID, decl_id=700025)
        assert comparable_types(exc, exc, table) is True

    def test_record_containing_exception_with_function_field_not_comparable(self) -> None:
        # A record field of exception type walks that exception's flattened
        # fields via the table, even though the record itself is table-resolved.
        table = TypeTable()
        handler_type = FunctionType(params=(), result=IntType())
        table.register(
            TypeDef(
                kind="exception",
                name="Failure",
                module_id=ENTRY_ID,
                fields=(("handler", handler_type),),
                decl_node_id=700025,
            )
        )
        exc = ExceptionType(name="Failure", module_id=ENTRY_ID, decl_id=700025)
        table.register(
            TypeDef(
                kind="record",
                name="Report",
                module_id=ENTRY_ID,
                fields=(("cause", exc),),
                decl_node_id=700026,
            )
        )
        handle = RecordType(name="Report", module_id=ENTRY_ID, decl_id=700026)
        assert comparable_types(handle, handle, table) is False

    def test_record_referencing_already_flagged_record_not_comparable(self) -> None:
        # X is unconditionally non-comparable (a function field); Y's only
        # field is a bare reference to X (not through an exception, unlike
        # the test above) — Y must inherit X's flag via the already-computed
        # fixpoint fact, not by re-walking X's own fields.
        table = TypeTable()
        fn_type = FunctionType(params=(), result=IntType())
        table.register(
            TypeDef(
                kind="record",
                name="X",
                module_id=ENTRY_ID,
                fields=(("fn", fn_type),),
                decl_node_id=700022,
            )
        )
        table.register(
            TypeDef(
                kind="record",
                name="Y",
                module_id=ENTRY_ID,
                fields=(("x", RecordType(name="X", module_id=ENTRY_ID, decl_id=700022)),),
                decl_node_id=700027,
            )
        )
        handle = RecordType(name="Y", module_id=ENTRY_ID, decl_id=700027)
        assert comparable_types(handle, handle, table) is False

    def test_dangling_field_reference_defaults_to_comparable(self) -> None:
        # Y's field references a declaration that was never registered (an
        # internal-invariant violation that should not happen for a
        # well-formed table); the fixpoint treats an unresolvable reference
        # as comparable rather than raising, defensively.
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Y",
                module_id=ENTRY_ID,
                fields=(("ghost", RecordType(name="Ghost", module_id=ENTRY_ID)),),
                decl_node_id=700027,
            )
        )
        handle = RecordType(name="Y", module_id=ENTRY_ID, decl_id=700027)
        assert comparable_types(handle, handle, table) is True

    def test_unregistered_handle_defaults_to_comparable(self) -> None:
        # Querying comparability of a handle whose own declaration was never
        # registered at all (as opposed to one merely referenced by a field)
        # is likewise defensive rather than a crash.
        table = TypeTable()
        handle = RecordType(name="Ghost", module_id=ENTRY_ID)
        assert comparable_types(handle, handle, table) is True

    def test_recursive_tree_is_comparable(self) -> None:
        # A self-referential enum (array/dict guard not even needed for
        # equality — only for inhabitation): the non-data-reachability fixpoint
        # must terminate on a cycle instead of recursing through the same
        # declaration's fields forever.
        table = TypeTable()
        register_typedef(
            table,
            enum_typedef(
                "Tree",
                {
                    "Leaf": {},
                    "Node": {
                        "value": IntType(),
                        "left": EnumType(name="Tree", module_id=ENTRY_ID, decl_id=700028),
                        "right": EnumType(name="Tree", module_id=ENTRY_ID, decl_id=700028),
                    },
                },
                decl_id=700028,
            ),
        )
        handle = EnumType(name="Tree", module_id=ENTRY_ID, decl_id=700028)
        assert comparable_types(handle, handle, table) is True

    def test_recursive_type_with_function_field_at_depth_not_comparable(self) -> None:
        # Same recursive shape as above, but one variant carries a function
        # field: the whole recursive type is non-comparable, exactly as a
        # non-recursive type containing a function field would be.
        table = TypeTable()
        handler_type = FunctionType(params=(), result=IntType())
        register_typedef(
            table,
            enum_typedef(
                "Tree",
                {
                    "Leaf": {},
                    "Handler": {"fn": handler_type},
                    "Node": {
                        "left": EnumType(name="Tree", module_id=ENTRY_ID, decl_id=700028),
                        "right": EnumType(name="Tree", module_id=ENTRY_ID, decl_id=700028),
                    },
                },
                decl_id=700028,
            ),
        )
        handle = EnumType(name="Tree", module_id=ENTRY_ID, decl_id=700028)
        assert comparable_types(handle, handle, table) is False

    def test_mutually_recursive_records_are_comparable(self) -> None:
        # A/B are mutually recursive through an array guard (inhabited) and
        # contain only scalar fields otherwise: both must be comparable, and
        # the fixpoint must not infinite-loop walking A -> B -> A -> ...
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="A",
                module_id=ENTRY_ID,
                fields=(
                    ("name", TextType()),
                    ("bs", ArrayType(RecordType(name="B", module_id=ENTRY_ID, decl_id=700015))),
                ),
                decl_node_id=700014,
            )
        )
        table.register(
            TypeDef(
                kind="record",
                name="B",
                module_id=ENTRY_ID,
                fields=(("a", RecordType(name="A", module_id=ENTRY_ID, decl_id=700014)),),
                decl_node_id=700015,
            )
        )
        a_handle = RecordType(name="A", module_id=ENTRY_ID, decl_id=700014)
        b_handle = RecordType(name="B", module_id=ENTRY_ID, decl_id=700015)
        assert comparable_types(a_handle, a_handle, table) is True
        assert comparable_types(b_handle, b_handle, table) is True


# ---------------------------------------------------------------------------
# The two consumers of the shared non-data-reachability fact
# ---------------------------------------------------------------------------


class TestNominalReachesNonData:
    def test_record_of_scalars_has_equality(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Point",
                module_id=ENTRY_ID,
                fields=(("x", IntType()), ("y", IntType())),
                decl_node_id=700000,
            )
        )
        handle = RecordType(name="Point", module_id=ENTRY_ID, decl_id=700000)
        assert table.nominal_reaches_non_data(handle) is False


class TestNominalIsJsonConvertible:
    def test_record_of_scalars_is_convertible(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Point",
                module_id=ENTRY_ID,
                fields=(("x", IntType()), ("y", IntType())),
                decl_node_id=700000,
            )
        )
        handle = RecordType(name="Point", module_id=ENTRY_ID, decl_id=700000)
        assert table.nominal_is_json_convertible(handle) is True


# ---------------------------------------------------------------------------
# The cast matrix — `cast_classification` and the `as json` rule it applies
# ---------------------------------------------------------------------------


def _bad_record_table() -> TypeTable:
    """A table with one JSON-convertible and one non-convertible record."""
    table = TypeTable()
    table.register(
        TypeDef(
            kind="record",
            name="Bad",
            module_id=ENTRY_ID,
            fields=(("a", FunctionType(params=(), result=UnitType())), ("x", IntType())),
            decl_node_id=700029,
        )
    )
    table.register(
        TypeDef(
            kind="record",
            name="Good",
            module_id=ENTRY_ID,
            fields=(("x", IntType()),),
            decl_node_id=700030,
        )
    )
    return table


class TestCastClassification:
    def test_bottom_source_is_assignable_to_cast_target(self) -> None:
        assert cast_classification(BottomType(), TextType(), TypeTable()) == CastKind.TOTAL_NOOP

    def test_text_to_text_noop(self) -> None:
        assert cast_classification(TextType(), TextType(), TypeTable()) == CastKind.TOTAL_NOOP

    def test_int_to_text_render(self) -> None:
        assert cast_classification(IntType(), TextType(), TypeTable()) == CastKind.TOTAL_RENDER

    def test_bool_to_text_render(self) -> None:
        assert cast_classification(BoolType(), TextType(), TypeTable()) == CastKind.TOTAL_RENDER

    def test_int_to_json_total_json(self) -> None:
        assert cast_classification(IntType(), JsonType(), TypeTable()) == CastKind.TOTAL_JSON

    def test_text_to_json_total_json(self) -> None:
        assert cast_classification(TextType(), JsonType(), TypeTable()) == CastKind.TOTAL_JSON

    def test_json_to_json_noop(self) -> None:
        assert cast_classification(JsonType(), JsonType(), TypeTable()) == CastKind.TOTAL_NOOP

    def test_text_to_int_fallible(self) -> None:
        assert cast_classification(TextType(), IntType(), TypeTable()) == CastKind.FALLIBLE

    def test_json_to_bool_fallible(self) -> None:
        assert cast_classification(JsonType(), BoolType(), TypeTable()) == CastKind.FALLIBLE

    def test_decimal_to_int_fallible(self) -> None:
        assert cast_classification(DecimalType(), IntType(), TypeTable()) == CastKind.FALLIBLE

    def test_int_to_decimal_noop(self) -> None:
        assert cast_classification(IntType(), DecimalType(), TypeTable()) == CastKind.TOTAL_NOOP

    def test_bool_to_int_static_error(self) -> None:
        assert cast_classification(BoolType(), IntType(), TypeTable()) == CastKind.STATIC_ERROR

    def test_int_to_bool_static_error(self) -> None:
        assert cast_classification(IntType(), BoolType(), TypeTable()) == CastKind.STATIC_ERROR

    def test_unit_source_static_error(self) -> None:
        assert cast_classification(UnitType(), TextType(), TypeTable()) == CastKind.STATIC_ERROR

    def test_array_of_scalars_to_json_total(self) -> None:
        # array[int] is JSON-shaped: not implicitly assignable to json (see
        # TestIsAssignable), but still convertible via an explicit cast.
        assert (
            cast_classification(ArrayType(elem=IntType()), JsonType(), TypeTable())
            == CastKind.TOTAL_JSON
        )

    def test_dict_of_scalars_to_json_total(self) -> None:
        assert (
            cast_classification(DictType(value=IntType()), JsonType(), TypeTable())
            == CastKind.TOTAL_JSON
        )

    def test_text_to_record_fallible(self) -> None:
        r = RecordType(name="R", decl_id=700035)
        assert cast_classification(TextType(), r, TypeTable()) == CastKind.FALLIBLE

    def test_json_to_record_fallible(self) -> None:
        r = RecordType(name="R", decl_id=700035)
        assert cast_classification(JsonType(), r, TypeTable()) == CastKind.FALLIBLE

    def test_exception_as_target_static_error(self) -> None:
        exc = ExceptionType(name="MyError")
        assert cast_classification(TextType(), exc, TypeTable()) == CastKind.STATIC_ERROR

    def test_exception_source_to_text_render(self) -> None:
        exc = ExceptionType(name="MyError")
        assert cast_classification(exc, TextType(), TypeTable()) == CastKind.TOTAL_RENDER

    def test_json_to_text_render(self) -> None:
        assert cast_classification(JsonType(), TextType(), TypeTable()) == CastKind.TOTAL_RENDER

    def test_int_to_array_static_error(self) -> None:
        assert (
            cast_classification(IntType(), ArrayType(IntType()), TypeTable())
            == CastKind.STATIC_ERROR
        )

    def test_record_to_json_total(self) -> None:
        table = _bad_record_table()
        good = RecordType(name="Good", module_id=ENTRY_ID, decl_id=700030)
        assert cast_classification(good, JsonType(), table) == CastKind.TOTAL_JSON

    def test_enum_to_json_total(self) -> None:
        table = TypeTable()
        register_typedef(table, enum_typedef("E", {"A": {}, "B": {"x": IntType()}}, decl_id=700031))
        assert (
            cast_classification(
                EnumType(name="E", module_id=ENTRY_ID, decl_id=700031), JsonType(), table
            )
            == CastKind.TOTAL_JSON
        )

    def test_exception_to_json_total(self) -> None:
        table = create_seeded_type_table()
        source = ExceptionType(name="Abort", module_id=RESERVED_ID)
        assert cast_classification(source, JsonType(), table) == CastKind.TOTAL_JSON

    def test_array_of_record_to_json_total(self) -> None:
        table = _bad_record_table()
        good = ArrayType(elem=RecordType(name="Good", module_id=ENTRY_ID, decl_id=700030))
        assert cast_classification(good, JsonType(), table) == CastKind.TOTAL_JSON

    def test_record_with_agent_field_to_json_static_error(self) -> None:
        table = _bad_record_table()
        bad = RecordType(name="Bad", module_id=ENTRY_ID, decl_id=700029)
        assert cast_classification(bad, JsonType(), table) == CastKind.STATIC_ERROR

    def test_array_of_record_with_agent_field_to_json_static_error(self) -> None:
        table = _bad_record_table()
        bad = ArrayType(elem=RecordType(name="Bad", module_id=ENTRY_ID, decl_id=700029))
        assert cast_classification(bad, JsonType(), table) == CastKind.STATIC_ERROR


class TestIsJsonConvertible:
    def test_host_minted_session_does_not_convert(self) -> None:
        session = BUILTIN_PRELUDE_TYPES["Session"]
        assert isinstance(session, RecordType)

        assert is_json_convertible(session, create_seeded_type_table()) is False

    def test_scalars_convert(self) -> None:
        table = TypeTable()
        for scalar in (TextType(), JsonType(), BoolType(), IntType(), DecimalType()):
            assert is_json_convertible(scalar, table) is True

    def test_nested_containers_follow_their_element_type(self) -> None:
        table = _bad_record_table()
        good = RecordType(name="Good", module_id=ENTRY_ID, decl_id=700030)
        bad = RecordType(name="Bad", module_id=ENTRY_ID, decl_id=700029)
        assert is_json_convertible(ArrayType(elem=ArrayType(elem=good)), table) is True
        assert is_json_convertible(DictType(value=ArrayType(elem=good)), table) is True
        assert is_json_convertible(ArrayType(elem=ArrayType(elem=bad)), table) is False
        assert is_json_convertible(DictType(value=ArrayType(elem=bad)), table) is False

    def test_recursive_declaration_converts(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Node",
                module_id=ENTRY_ID,
                fields=(
                    ("tag", IntType()),
                    (
                        "children",
                        ArrayType(elem=RecordType(name="Node", module_id=ENTRY_ID, decl_id=700032)),
                    ),
                ),
                decl_node_id=700032,
            )
        )
        node = RecordType(name="Node", module_id=ENTRY_ID, decl_id=700032)
        assert is_json_convertible(node, table) is True

    def test_free_type_variable_never_converts(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Box",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(("value", TypeVarType("T")),),
                decl_node_id=700002,
            )
        )
        assert is_json_convertible(TypeVarType("T"), table) is False
        assert is_json_convertible(ArrayType(elem=TypeVarType("T")), table) is False
        boxed = RecordType(
            name="Box", type_args=(TypeVarType("T"),), module_id=ENTRY_ID, decl_id=700002
        )
        assert is_json_convertible(boxed, table) is False
        concrete = RecordType(
            name="Box", type_args=(IntType(),), module_id=ENTRY_ID, decl_id=700002
        )
        assert is_json_convertible(concrete, table) is True


class TestJsonCastHint:
    """The hint names ``as json`` only where that cast is what the value needs."""

    def test_container_and_nominal_into_json_are_hinted(self) -> None:
        table = _bad_record_table()
        good = RecordType(name="Good", module_id=ENTRY_ID, decl_id=700030)
        for value_type in (ArrayType(elem=IntType()), DictType(value=IntType()), good):
            assert "as json" in json_cast_hint(value_type, JsonType(), table)

    def test_nonconvertible_container_into_json_is_not_hinted(self) -> None:
        """A cast hint is not offered when no JSON cast can make the value valid."""
        table = _bad_record_table()
        bad = RecordType(name="Bad", module_id=ENTRY_ID, decl_id=700029)
        assert json_cast_hint(ArrayType(elem=bad), JsonType(), table) == ""

    def test_scalar_into_json_is_not_hinted(self) -> None:
        # A scalar is absorbed into a json slot implicitly, so naming the
        # explicit cast would point at a step the value does not need.
        table = TypeTable()
        for scalar in (TextType(), BoolType(), IntType(), DecimalType(), JsonType()):
            assert json_cast_hint(scalar, JsonType(), table) == ""

    def test_json_into_another_type_is_not_hinted(self) -> None:
        # The reverse direction: the fix is a cast to the target type, never
        # `as json`.
        table = TypeTable()
        for target in (TextType(), ArrayType(elem=IntType()), DictType(value=IntType())):
            assert json_cast_hint(JsonType(), target, table) == ""


class TestJsonRepresentationObstacle:
    def test_convertible_type_has_no_obstacle(self) -> None:
        table = _bad_record_table()
        good = ArrayType(elem=RecordType(name="Good", module_id=ENTRY_ID, decl_id=700030))
        assert table.json_representation_obstacle(good) is None

    def test_structural_non_data_leaf_through_a_dict_is_named(self) -> None:
        table = TypeTable()
        message = table.json_representation_obstacle(DictType(value=ArrayType(elem=UnitType())))
        assert message is not None
        assert "unit" in message

    def test_culprit_is_reported_through_a_nested_declaration(self) -> None:
        table = _bad_record_table()
        table.register(
            TypeDef(
                kind="record",
                name="Outer",
                module_id=ENTRY_ID,
                fields=(
                    ("label", TextType()),
                    (
                        "inner",
                        ArrayType(elem=RecordType(name="Bad", module_id=ENTRY_ID, decl_id=700029)),
                    ),
                ),
                decl_node_id=700033,
            )
        )
        outer = RecordType(name="Outer", module_id=ENTRY_ID, decl_id=700033)
        message = table.json_representation_obstacle(outer)
        assert message is not None
        assert "'a'" in message
        assert "Bad" in message

    def test_culprit_search_visits_a_shared_declaration_once(self) -> None:
        # Two independent paths reach Mid, whose own fields are all clean
        # (a scalar and a convertible nominal) — the blame lies one hop
        # further on. The search must converge rather than re-expand Mid.
        table = _bad_record_table()
        bad = RecordType(name="Bad", module_id=ENTRY_ID, decl_id=700029)
        good = RecordType(name="Good", module_id=ENTRY_ID, decl_id=700030)
        mid = RecordType(name="Mid", module_id=ENTRY_ID, decl_id=700009)
        table.register(
            TypeDef(
                kind="record",
                name="Mid",
                module_id=ENTRY_ID,
                fields=(("x", IntType()), ("ok", good), ("deep", bad)),
                decl_node_id=700009,
            )
        )
        side_ids = {"Left": 700040, "Right": 700041}
        for side, side_id in side_ids.items():
            table.register(
                TypeDef(
                    kind="record",
                    name=side,
                    module_id=ENTRY_ID,
                    fields=(("m", mid),),
                    decl_node_id=side_id,
                )
            )
        table.register(
            TypeDef(
                kind="record",
                name="Top",
                module_id=ENTRY_ID,
                fields=(
                    ("l", RecordType(name="Left", module_id=ENTRY_ID, decl_id=side_ids["Left"])),
                    ("r", RecordType(name="Right", module_id=ENTRY_ID, decl_id=side_ids["Right"])),
                ),
                decl_node_id=700034,
            )
        )
        message = table.json_representation_obstacle(
            RecordType(name="Top", module_id=ENTRY_ID, decl_id=700034)
        )
        assert message is not None
        assert "'a'" in message
        assert "Bad" in message

    def test_enum_variant_field_is_named(self) -> None:
        table = TypeTable()
        register_typedef(
            table,
            enum_typedef(
                "Holder",
                {"Empty": {}, "Full": {"run": FunctionType(params=(), result=IntType())}},
                decl_id=700023,
            ),
        )
        holder = EnumType(name="Holder", module_id=ENTRY_ID, decl_id=700023)
        message = table.json_representation_obstacle(holder)
        assert message is not None
        assert "'run'" in message
        assert "Holder" in message

    def test_exception_descendant_poisons_its_ancestor(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="Base",
                module_id=ENTRY_ID,
                fields=(("code", IntType()),),
                decl_node_id=700013,
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Child",
                module_id=ENTRY_ID,
                base=700013,
                fields=(("handler", FunctionType(params=(IntType(),), result=IntType())),),
                decl_node_id=700012,
            )
        )
        base = ExceptionType(name="Base", module_id=ENTRY_ID, decl_id=700013)
        assert is_json_convertible(base, table) is False
        message = table.json_representation_obstacle(base)
        assert message is not None
        assert "'handler'" in message
        assert "Child" in message

    def test_type_variable_is_named_when_nothing_else_is_to_blame(self) -> None:
        table = TypeTable()
        message = table.json_representation_obstacle(ArrayType(elem=TypeVarType("T")))
        assert message is not None
        assert "'T'" in message

    def test_unresolved_inference_variable_has_no_specific_obstacle(self) -> None:
        table = TypeTable()
        assert table.json_representation_obstacle(ArrayType(elem=InferenceVarType(1))) is None


# ---------------------------------------------------------------------------
# Finiteness (instantiation-closure) analysis
# ---------------------------------------------------------------------------


_PAIR_DECL_ID = 700200


def _pair_def(name: str = "Pair") -> TypeDef:
    """A plain non-recursive generic record with two independent parameters."""
    return TypeDef(
        kind="record",
        name=name,
        module_id=ENTRY_ID,
        type_params=("X", "Y"),
        fields=(("x", TypeVarType("X")), ("y", TypeVarType("Y"))),
        decl_node_id=_PAIR_DECL_ID,
    )


class TestInhabitationAnalysis:
    def test_referenced_uninhabitable_member_is_rejected_by_program_checking(self) -> None:
        with pytest.raises(AglTypeError):
            _check("record Bad(next: Bad)\nenum E = ::Bad | Good\n()")

    def test_referenced_enum_member_still_requires_its_own_finite_value(self) -> None:
        table = TypeTable()
        bad = RecordType("Bad", module_id=ENTRY_ID, decl_id=700036)
        good = RecordType("Good", module_id=ENTRY_ID, decl_id=700037)
        table.register(
            TypeDef(
                kind="record",
                name="Bad",
                module_id=ENTRY_ID,
                fields=(("next", bad),),
                decl_node_id=bad.decl_id,
            )
        )
        table.register(
            TypeDef(
                kind="record",
                name="Good",
                module_id=ENTRY_ID,
                decl_node_id=good.decl_id,
            )
        )
        table.register(
            TypeDef(
                kind="enum",
                name="E",
                module_id=ENTRY_ID,
                members=(bad, good),
                decl_node_id=700038,
            )
        )

        assert compute_uninhabited(table) == frozenset({bad.decl_id})

    def test_dangling_nominal_reference_stays_uninhabited(self) -> None:
        """A malformed table with a missing target does not mark the source inhabited."""
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="R",
                module_id=ENTRY_ID,
                fields=(("missing", RecordType("Missing", module_id=ENTRY_ID)),),
                decl_node_id=700035,
            )
        )

        assert compute_uninhabited(table) == frozenset({700035})

    def test_exception_with_missing_base_stays_uninhabited(self) -> None:
        """A malformed exception base link is not treated as constructible evidence."""
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="E",
                module_id=ENTRY_ID,
                base=999_999_999,  # never registered: a dangling base identity
                decl_node_id=700031,
            )
        )

        assert compute_uninhabited(table) == frozenset({700031})


class TestFiniteClosure:
    def test_nominal_references_walks_nested_type_shapes(self) -> None:
        exc = ExceptionType("Oops", module_id=ENTRY_ID)
        enum = EnumType("Choice", module_id=ENTRY_ID)
        box = RecordType("Box", type_args=(enum,), module_id=ENTRY_ID, decl_id=700002)
        typ = FunctionType(params=(ArrayType(exc),), result=DictType(box))
        assert list(nominal_references(typ)) == [exc, box, enum]
        assert list(nominal_references(BoolType())) == []

    def test_uniform_self_reference_is_finite(self) -> None:
        # Tree[T] referencing Tree[T]: the parameter-dependency self-loop
        # (T -> T) passes the WHOLE argument through unchanged — never a
        # proper subterm — so it is not growing.
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Tree",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "children",
                        ArrayType(
                            RecordType(
                                "Tree",
                                type_args=(TypeVarType("T"),),
                                module_id=ENTRY_ID,
                                decl_id=700028,
                            )
                        ),
                    ),
                ),
                decl_node_id=700028,
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "Tree") is True

    def test_argument_constant_reference_is_finite(self) -> None:
        # R[int] referenced from R[T]'s own body: the argument template
        # "int" contains none of R's own parameters at all, so there is no
        # parameter-dependency edge whatsoever for this reference.
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="R",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "constref",
                        RecordType("R", type_args=(IntType(),), module_id=ENTRY_ID, decl_id=700035),
                    ),
                ),
                decl_node_id=700035,
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "R") is True

    def test_permutation_cycle_is_finite(self) -> None:
        # Swap[A, B] referencing Swap[B, A]: each parameter is passed through
        # to a DIFFERENT slot unchanged (never a proper subterm), so the
        # A -> B -> A parameter cycle has no growing edge.
        table = TypeTable()
        register_typedef(
            table,
            enum_typedef(
                "Swap",
                {
                    "Base": {"a": TypeVarType("A"), "b": TypeVarType("B")},
                    "Rec": {
                        "inner": EnumType(
                            "Swap",
                            type_args=(TypeVarType("B"), TypeVarType("A")),
                            module_id=ENTRY_ID,
                            decl_id=700036,
                        )
                    },
                },
                type_params=("A", "B"),
                decl_id=700036,
            ),
        )
        assert table.has_finite_closure(ENTRY_ID, "Swap") is True

    def test_growing_via_nominal_argument_is_infinite(self) -> None:
        # Perfect[T] referencing Perfect[Pair[T, T]]: T occurs nested inside
        # Pair's own argument list, a proper subterm of the whole argument
        # template — a growing self-loop.
        table = TypeTable()
        table.register(_pair_def())
        table.register(
            TypeDef(
                kind="record",
                name="Perfect",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "next",
                        RecordType(
                            "Perfect",
                            type_args=(
                                RecordType(
                                    "Pair",
                                    type_args=(TypeVarType("T"), TypeVarType("T")),
                                    module_id=ENTRY_ID,
                                    decl_id=700200,
                                ),
                            ),
                            module_id=ENTRY_ID,
                            decl_id=700037,
                        ),
                    ),
                ),
                decl_node_id=700037,
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "Perfect") is False
        # Pair itself is unrelated (non-recursive) and stays finite.
        assert table.has_finite_closure(ENTRY_ID, "Pair") is True

    def test_growing_via_array_is_infinite(self) -> None:
        # P[T] referencing P[array[T]]: T occurs under the array constructor,
        # a proper subterm of the argument template.
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="P",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "next",
                        RecordType(
                            "P",
                            type_args=(ArrayType(TypeVarType("T")),),
                            module_id=ENTRY_ID,
                            decl_id=700038,
                        ),
                    ),
                ),
                decl_node_id=700038,
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "P") is False

    def test_mutual_growing_across_two_declarations_is_infinite(self) -> None:
        # A[T] references B[Pair[T, T]]; B[T] references A[T] — the growing
        # step and the cycle-closing step live on DIFFERENT declarations.
        table = TypeTable()
        table.register(_pair_def())
        table.register(
            TypeDef(
                kind="record",
                name="A",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "b",
                        RecordType(
                            "B",
                            type_args=(
                                RecordType(
                                    "Pair",
                                    type_args=(TypeVarType("T"), TypeVarType("T")),
                                    module_id=ENTRY_ID,
                                    decl_id=700200,
                                ),
                            ),
                            module_id=ENTRY_ID,
                            decl_id=700015,
                        ),
                    ),
                ),
                decl_node_id=700014,
            )
        )
        table.register(
            TypeDef(
                kind="record",
                name="B",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "a",
                        RecordType(
                            "A", type_args=(TypeVarType("T"),), module_id=ENTRY_ID, decl_id=700014
                        ),
                    ),
                ),
                decl_node_id=700015,
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "A") is False
        assert table.has_finite_closure(ENTRY_ID, "B") is False

    def test_growing_edge_in_one_scc_member_poisons_whole_scc(self) -> None:
        # C[T] references D[T] (uniform, non-growing); D[T] references
        # C[Pair[T, T]] (growing). C and D form one SCC; the growing edge on
        # the D -> C leg is enough to mark BOTH C and D infinite.
        table = TypeTable()
        table.register(_pair_def())
        table.register(
            TypeDef(
                kind="record",
                name="C",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "d",
                        RecordType(
                            "D", type_args=(TypeVarType("T"),), module_id=ENTRY_ID, decl_id=700040
                        ),
                    ),
                ),
                decl_node_id=700039,
            )
        )
        table.register(
            TypeDef(
                kind="record",
                name="D",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "c",
                        RecordType(
                            "C",
                            type_args=(
                                RecordType(
                                    "Pair",
                                    type_args=(TypeVarType("T"), TypeVarType("T")),
                                    module_id=ENTRY_ID,
                                    decl_id=700200,
                                ),
                            ),
                            module_id=ENTRY_ID,
                            decl_id=700039,
                        ),
                    ),
                ),
                decl_node_id=700040,
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "C") is False
        assert table.has_finite_closure(ENTRY_ID, "D") is False

    def test_non_recursive_generic_is_finite(self) -> None:
        table = TypeTable()
        table.register(_pair_def())
        assert table.has_finite_closure(ENTRY_ID, "Pair") is True

    def test_non_generic_exception_is_finite(self) -> None:
        # Exceptions are never generic, so they contribute no
        # parameter-dependency nodes at all — any recursive exception chain
        # (guarded through an array/dict field, as inhabitation requires) is
        # unconditionally finite.
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="Chain",
                module_id=ENTRY_ID,
                fields=(
                    ("code", IntType()),
                    (
                        "causes",
                        ArrayType(ExceptionType("Chain", module_id=ENTRY_ID, decl_id=700041)),
                    ),
                ),
                decl_node_id=700041,
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "Chain") is True

    def test_unregistered_declaration_defaults_to_finite(self) -> None:
        table = TypeTable()
        assert table.has_finite_closure(ENTRY_ID, "Ghost") is True

    def test_has_finite_schema_reports_infinite_for_nested_perfect_field(self) -> None:
        # A non-recursive record containing a Perfect[int] field: the
        # reachability query must walk INTO the field's own type_args (not
        # just the record's direct fields) to find the infinite declaration.
        table = TypeTable()
        table.register(_pair_def())
        table.register(
            TypeDef(
                kind="record",
                name="Perfect",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "next",
                        RecordType(
                            "Perfect",
                            type_args=(
                                RecordType(
                                    "Pair",
                                    type_args=(TypeVarType("T"), TypeVarType("T")),
                                    module_id=ENTRY_ID,
                                    decl_id=700200,
                                ),
                            ),
                            module_id=ENTRY_ID,
                            decl_id=700037,
                        ),
                    ),
                ),
                decl_node_id=700037,
            )
        )
        table.register(
            TypeDef(
                kind="record",
                name="Holder",
                module_id=ENTRY_ID,
                fields=(
                    (
                        "p",
                        RecordType(
                            "Perfect", type_args=(IntType(),), module_id=ENTRY_ID, decl_id=700037
                        ),
                    ),
                ),
                decl_node_id=700023,
            )
        )
        holder = RecordType("Holder", module_id=ENTRY_ID, decl_id=700023)
        assert table.has_finite_schema(holder) is False

    def test_has_finite_schema_ignores_phantom_type_arguments(self) -> None:
        table = TypeTable()
        table.register(_pair_def())
        table.register(
            TypeDef(
                kind="record",
                name="Perfect",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "next",
                        RecordType(
                            "Perfect",
                            type_args=(
                                RecordType(
                                    "Pair",
                                    type_args=(TypeVarType("T"), TypeVarType("T")),
                                    module_id=ENTRY_ID,
                                    decl_id=700200,
                                ),
                            ),
                            module_id=ENTRY_ID,
                            decl_id=700037,
                        ),
                    ),
                ),
                decl_node_id=700037,
            )
        )
        table.register(
            TypeDef(
                kind="record",
                name="Phantom",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(),
                decl_node_id=700042,
            )
        )
        phantom = RecordType(
            "Phantom",
            type_args=(
                RecordType("Perfect", type_args=(IntType(),), module_id=ENTRY_ID, decl_id=700037),
            ),
            module_id=ENTRY_ID,
            decl_id=700042,
        )
        assert table.has_finite_schema(phantom) is True

    def test_phantom_recursive_argument_growth_is_finite_schema(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="R",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    (
                        "children",
                        ArrayType(
                            RecordType(
                                "R",
                                type_args=(ArrayType(TypeVarType("T")),),
                                module_id=ENTRY_ID,
                                decl_id=700035,
                            )
                        ),
                    ),
                ),
                decl_node_id=700035,
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "R") is True
        assert (
            table.has_finite_schema(
                RecordType("R", type_args=(IntType(),), module_id=ENTRY_ID, decl_id=700035)
            )
            is True
        )

    def test_nested_phantom_argument_growth_is_finite_schema(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Phantom",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(),
                decl_node_id=700042,
            )
        )
        register_typedef(
            table,
            enum_typedef(
                "E",
                {
                    "Leaf": {"value": TypeVarType("T")},
                    "Node": {
                        "child": EnumType(
                            "E",
                            type_args=(
                                RecordType(
                                    "Phantom",
                                    type_args=(ArrayType(TypeVarType("T")),),
                                    module_id=ENTRY_ID,
                                    decl_id=700042,
                                ),
                            ),
                            module_id=ENTRY_ID,
                            decl_id=700031,
                        )
                    },
                },
                type_params=("T",),
                decl_id=700031,
            ),
        )

        assert table.has_finite_closure(ENTRY_ID, "E") is True
        assert (
            table.has_finite_schema(
                EnumType("E", type_args=(IntType(),), module_id=ENTRY_ID, decl_id=700031)
            )
            is True
        )

    def test_unknown_nested_nominal_argument_counts_as_schema_growth(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="R",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "child",
                        RecordType(
                            "R",
                            type_args=(
                                RecordType(
                                    "Unknown",
                                    type_args=(ArrayType(TypeVarType("T")),),
                                    module_id=ENTRY_ID,
                                ),
                            ),
                            module_id=ENTRY_ID,
                            decl_id=700035,
                        ),
                    ),
                ),
                decl_node_id=700035,
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "R") is False

    def test_extra_nested_nominal_argument_counts_as_schema_growth(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Box",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(("value", TypeVarType("T")),),
                decl_node_id=700002,
            )
        )
        table.register(
            TypeDef(
                kind="record",
                name="R",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "child",
                        RecordType(
                            "R",
                            type_args=(
                                RecordType(
                                    "Box",
                                    type_args=(TypeVarType("T"), ArrayType(TypeVarType("T"))),
                                    module_id=ENTRY_ID,
                                    decl_id=700002,
                                ),
                            ),
                            module_id=ENTRY_ID,
                            decl_id=700035,
                        ),
                    ),
                ),
                decl_node_id=700035,
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "R") is False

    def test_has_finite_schema_reports_finite_for_nested_tree_field(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Tree",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "children",
                        ArrayType(
                            RecordType(
                                "Tree",
                                type_args=(TypeVarType("T"),),
                                module_id=ENTRY_ID,
                                decl_id=700028,
                            )
                        ),
                    ),
                ),
                decl_node_id=700028,
            )
        )
        table.register(
            TypeDef(
                kind="record",
                name="Holder",
                module_id=ENTRY_ID,
                fields=(
                    (
                        "t",
                        RecordType(
                            "Tree", type_args=(IntType(),), module_id=ENTRY_ID, decl_id=700028
                        ),
                    ),
                ),
                decl_node_id=700023,
            )
        )
        holder = RecordType("Holder", module_id=ENTRY_ID, decl_id=700023)
        assert table.has_finite_schema(holder) is True

    def test_has_finite_schema_true_for_scalar_root(self) -> None:
        table = TypeTable()
        assert table.has_finite_schema(IntType()) is True

    def test_compute_finite_closure_successors_reach_transitively(self) -> None:
        # Direct check of compute_finite_closure's own result shape: Holder
        # does not itself reference Perfect's declaration in its OWN
        # closure computation membership, but its successors chain through
        # to Perfect via the field's nominal reference — exercised here via
        # the function directly rather than through has_finite_schema.
        table = TypeTable()
        table.register(_pair_def())
        table.register(
            TypeDef(
                kind="record",
                name="Perfect",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "next",
                        RecordType(
                            "Perfect",
                            type_args=(
                                RecordType(
                                    "Pair",
                                    type_args=(TypeVarType("T"), TypeVarType("T")),
                                    module_id=ENTRY_ID,
                                    decl_id=700200,
                                ),
                            ),
                            module_id=ENTRY_ID,
                            decl_id=700037,
                        ),
                    ),
                ),
                decl_node_id=700037,
            )
        )
        result = compute_finite_closure(table)
        perfect_id = 700037
        pair_id = 700200
        assert perfect_id in result.infinite
        assert result.successors[perfect_id] == frozenset({perfect_id, pair_id})

    def test_caches_result_and_invalidates_on_a_new_declaration(self) -> None:
        # A redeclaration always mints a fresh identity (the old one is
        # retained, never mutated in place -- see ``TypeTable.register``), so
        # the cache must be invalidated by an ordinary new registration, not
        # merely by a change under an existing identity.
        table = TypeTable()
        table.register(_pair_def())
        assert table.has_finite_closure(ENTRY_ID, "Pair") is True
        table.register(
            TypeDef(
                kind="record",
                name="Pair",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "next",
                        RecordType(
                            "Pair",
                            type_args=(ArrayType(TypeVarType("T")),),
                            module_id=ENTRY_ID,
                            decl_id=700201,
                        ),
                    ),
                ),
                decl_node_id=700201,
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "Pair") is False

    def test_growing_via_dict_is_infinite(self) -> None:
        # Q[T] referencing Q[dict[T]]: T occurs under the dict constructor,
        # a proper subterm of the argument template.
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Q",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "next",
                        RecordType(
                            "Q",
                            type_args=(DictType(TypeVarType("T")),),
                            module_id=ENTRY_ID,
                            decl_id=700043,
                        ),
                    ),
                ),
                decl_node_id=700043,
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "Q") is False

    def test_growing_via_function_type_is_infinite(self) -> None:
        # F[T] referencing F[(T) -> T]: T occurs under the function
        # constructor, a proper subterm of the argument template.
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="F",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "next",
                        RecordType(
                            "F",
                            type_args=(
                                FunctionType(params=(TypeVarType("T"),), result=TypeVarType("T")),
                            ),
                            module_id=ENTRY_ID,
                            decl_id=700044,
                        ),
                    ),
                ),
                decl_node_id=700044,
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "F") is False

    def test_exception_with_base_contributes_no_parameter_edges(self) -> None:
        # The `extends` base is a reference edge like any other, but
        # exceptions are never generic, so it never contributes a
        # parameter-dependency node — the base chain stays finite.
        table = TypeTable()
        table.register(
            TypeDef(kind="exception", name="Root", module_id=ENTRY_ID, decl_node_id=700008)
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Derived",
                module_id=ENTRY_ID,
                fields=(("code", IntType()),),
                base=700008,
                decl_node_id=700045,
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "Root") is True
        assert table.has_finite_closure(ENTRY_ID, "Derived") is True

    def test_dangling_reference_defaults_to_finite(self) -> None:
        # A field referencing a declaration that was never registered (same
        # defensive scenario as the non-data-reachability fixpoint): the
        # dangling reference must not crash finiteness analysis, and
        # defaults permissively to finite.
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Y",
                module_id=ENTRY_ID,
                fields=(("ghost", RecordType("Ghost", module_id=ENTRY_ID)),),
                decl_node_id=700027,
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "Y") is True

    def test_schema_canonical_type_preserves_unregistered_reference_args(self) -> None:
        table = TypeTable()
        ghost = RecordType(
            "Ghost",
            type_args=(ArrayType(IntType()),),
            module_id=ENTRY_ID,
        )
        assert table.canonical_schema_type(ghost) == ghost
        assert table.schema_relevant_type_args(ghost) == (ArrayType(IntType()),)

    def test_schema_canonical_type_preserves_extra_defensive_args(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Weird",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(("value", TypeVarType("T")),),
                decl_node_id=700046,
            )
        )
        weird = RecordType(
            "Weird", type_args=(IntType(), TextType()), module_id=ENTRY_ID, decl_id=700046
        )
        assert table.canonical_schema_type(weird) == weird
        assert table.schema_relevant_type_args(weird) == (IntType(), TextType())

    def test_argument_template_type_var_foreign_to_source_is_ignored(self) -> None:
        # Defensive: an argument template's type variable that is not among
        # the REFERENCING declaration's own type parameters (a malformed
        # template — never produced by the real type builder) is simply
        # ignored rather than crashing, mirroring the non-data-reachability
        # fixpoint's "ignore what does not fit the expected shape" stance.
        table = TypeTable()
        table.register(_pair_def())
        table.register(
            TypeDef(
                kind="record",
                name="A",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "b",
                        RecordType(
                            "B", type_args=(TypeVarType("Q"),), module_id=ENTRY_ID, decl_id=700015
                        ),
                    ),
                ),
                decl_node_id=700014,
            )
        )
        table.register(
            TypeDef(
                kind="record",
                name="B",
                module_id=ENTRY_ID,
                type_params=("Q",),
                fields=(
                    ("value", TypeVarType("Q")),
                    (
                        "a",
                        RecordType(
                            "A", type_args=(TypeVarType("Q"),), module_id=ENTRY_ID, decl_id=700014
                        ),
                    ),
                ),
                decl_node_id=700015,
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "A") is True
        assert table.has_finite_closure(ENTRY_ID, "B") is True

    def test_has_finite_schema_walks_into_root_type_args(self) -> None:
        # The reachability query's initial walk must look INSIDE the root
        # handle's own type_args, not just at the root handle itself: here
        # the infinite `Perfect` declaration is nested as a type ARGUMENT of
        # a (finite, unrelated) generic `Box[T]` root.
        table = TypeTable()
        table.register(_pair_def())
        table.register(
            TypeDef(
                kind="record",
                name="Perfect",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "next",
                        RecordType(
                            "Perfect",
                            type_args=(
                                RecordType(
                                    "Pair",
                                    type_args=(TypeVarType("T"), TypeVarType("T")),
                                    module_id=ENTRY_ID,
                                    decl_id=700200,
                                ),
                            ),
                            module_id=ENTRY_ID,
                            decl_id=700037,
                        ),
                    ),
                ),
                decl_node_id=700037,
            )
        )
        table.register(
            TypeDef(
                kind="record",
                name="Box",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(("value", TypeVarType("T")),),
                decl_node_id=700002,
            )
        )
        root = RecordType(
            "Box",
            type_args=(
                RecordType("Perfect", type_args=(IntType(),), module_id=ENTRY_ID, decl_id=700037),
            ),
            module_id=ENTRY_ID,
            decl_id=700002,
        )
        assert table.has_finite_schema(root) is False

    def test_has_finite_schema_walks_function_type_root(self) -> None:
        # A bare FunctionType root whose result carries the infinite
        # Perfect[int] instantiation.
        table = TypeTable()
        table.register(_pair_def())
        table.register(
            TypeDef(
                kind="record",
                name="Perfect",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "next",
                        RecordType(
                            "Perfect",
                            type_args=(
                                RecordType(
                                    "Pair",
                                    type_args=(TypeVarType("T"), TypeVarType("T")),
                                    module_id=ENTRY_ID,
                                    decl_id=700200,
                                ),
                            ),
                            module_id=ENTRY_ID,
                            decl_id=700037,
                        ),
                    ),
                ),
                decl_node_id=700037,
            )
        )
        root = FunctionType(
            params=(IntType(),),
            result=RecordType(
                "Perfect", type_args=(IntType(),), module_id=ENTRY_ID, decl_id=700037
            ),
        )
        assert table.has_finite_schema(root) is False

    def test_has_finite_schema_dedupes_repeated_declaration_in_root(self) -> None:
        # X appears TWICE as a sibling type argument of the (finite) root —
        # the reachability walk must not re-process an already-seen
        # declaration key a second time.
        table = TypeTable()
        table.register(_pair_def())
        table.register(
            TypeDef(
                kind="record",
                name="X",
                module_id=ENTRY_ID,
                fields=(("v", IntType()),),
                decl_node_id=700022,
            )
        )
        x_handle = RecordType("X", module_id=ENTRY_ID, decl_id=700022)
        root = RecordType(
            "Pair", type_args=(x_handle, x_handle), module_id=ENTRY_ID, decl_id=700200
        )
        assert table.has_finite_schema(root) is True

    # -----------------------------------------------------------------
    # first_infinite_declaration / no_finite_schema_message
    # -----------------------------------------------------------------

    def _perfect_table(self) -> TypeTable:
        """A table with ``Pair[A, B]`` and the growing ``Perfect[T]`` declaration."""
        table = TypeTable()
        table.register(_pair_def())
        table.register(
            TypeDef(
                kind="record",
                name="Perfect",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "next",
                        RecordType(
                            "Perfect",
                            type_args=(
                                RecordType(
                                    "Pair",
                                    type_args=(TypeVarType("T"), TypeVarType("T")),
                                    module_id=ENTRY_ID,
                                    decl_id=700200,
                                ),
                            ),
                            module_id=ENTRY_ID,
                            decl_id=700037,
                        ),
                    ),
                ),
                decl_node_id=700037,
            )
        )
        return table

    def test_first_infinite_declaration_is_none_for_finite_root(self) -> None:
        table = self._perfect_table()
        pair_int = RecordType(
            "Pair", type_args=(IntType(), IntType()), module_id=ENTRY_ID, decl_id=700200
        )
        assert table.first_infinite_declaration(IntType()) is None
        assert table.first_infinite_declaration(pair_int) is None

    def test_first_infinite_declaration_names_root_itself(self) -> None:
        table = self._perfect_table()
        perfect_int = RecordType(
            "Perfect", type_args=(IntType(),), module_id=ENTRY_ID, decl_id=700037
        )
        culprit = table.first_infinite_declaration(perfect_int)
        assert culprit is not None
        assert (culprit.module_id, culprit.scope_path, culprit.name) == (ENTRY_ID, (), "Perfect")

    def test_first_infinite_declaration_names_culprit_reached_through_field(self) -> None:
        table = self._perfect_table()
        table.register(
            TypeDef(
                kind="record",
                name="Holder",
                module_id=ENTRY_ID,
                fields=(
                    (
                        "p",
                        RecordType(
                            "Perfect", type_args=(IntType(),), module_id=ENTRY_ID, decl_id=700037
                        ),
                    ),
                ),
                decl_node_id=700023,
            )
        )
        holder = RecordType("Holder", module_id=ENTRY_ID, decl_id=700023)
        culprit = table.first_infinite_declaration(holder)
        assert culprit is not None
        assert (culprit.module_id, culprit.scope_path, culprit.name) == (ENTRY_ID, (), "Perfect")

    def test_no_finite_schema_message_is_none_for_finite_type(self) -> None:
        table = self._perfect_table()
        pair_int = RecordType(
            "Pair", type_args=(IntType(), IntType()), module_id=ENTRY_ID, decl_id=700200
        )
        assert table.no_finite_schema_message(pair_int, use="a cast target") is None

    def test_no_finite_schema_message_names_root_when_root_is_the_culprit(self) -> None:
        table = self._perfect_table()
        perfect_int = RecordType(
            "Perfect", type_args=(IntType(),), module_id=ENTRY_ID, decl_id=700037
        )
        message = table.no_finite_schema_message(perfect_int, use="an agent output type")
        assert message is not None
        assert "Perfect[int]" in message
        assert "an agent output type" in message
        assert "no finite JSON schema" in message

    def test_no_finite_schema_message_mentions_both_root_and_culprit(self) -> None:
        table = self._perfect_table()
        table.register(
            TypeDef(
                kind="record",
                name="Holder",
                module_id=ENTRY_ID,
                fields=(
                    (
                        "p",
                        RecordType(
                            "Perfect", type_args=(IntType(),), module_id=ENTRY_ID, decl_id=700037
                        ),
                    ),
                ),
                decl_node_id=700023,
            )
        )
        holder = RecordType("Holder", module_id=ENTRY_ID, decl_id=700023)
        message = table.no_finite_schema_message(holder, use="a parameter type")
        assert message is not None
        assert "Holder" in message
        assert "Perfect" in message
        assert "a parameter type" in message

    def test_no_finite_schema_message_qualifies_culprit_in_non_entry_module(self) -> None:
        # The culprit ("Perfect") is reached through a field, not the root
        # ("Holder") itself, and lives in a non-entry module: it must be
        # named with its module qualifier (matching the ``mod::Name``
        # convention RecordType/EnumType's own __repr__ uses) so it is never
        # ambiguous with a same-named declaration elsewhere.
        mod = ModuleId.from_path("mod_a")
        table = TypeTable()
        table.register(_pair_def())
        table.register(
            TypeDef(
                kind="record",
                name="Perfect",
                module_id=mod,
                type_params=("T",),
                fields=(
                    ("value", TypeVarType("T")),
                    (
                        "next",
                        RecordType(
                            "Perfect",
                            type_args=(
                                RecordType(
                                    "Pair",
                                    type_args=(TypeVarType("T"), TypeVarType("T")),
                                    module_id=ENTRY_ID,
                                    decl_id=700200,
                                ),
                            ),
                            module_id=mod,
                            decl_id=700047,
                        ),
                    ),
                ),
                decl_node_id=700047,
            )
        )
        table.register(
            TypeDef(
                kind="record",
                name="Holder",
                module_id=ENTRY_ID,
                fields=(
                    (
                        "p",
                        RecordType(
                            "Perfect", type_args=(IntType(),), module_id=mod, decl_id=700047
                        ),
                    ),
                ),
                decl_node_id=700023,
            )
        )
        holder = RecordType("Holder", module_id=ENTRY_ID, decl_id=700023)
        message = table.no_finite_schema_message(holder, use="a parameter type")
        assert message is not None
        assert "mod_a::Perfect" in message
        assert "a parameter type" in message


# ---------------------------------------------------------------------------
# Declaration-keyed identity: RecordType/EnumType/ExceptionType.decl_id and
# TypeDef.decl_node_id
# ---------------------------------------------------------------------------


class TestDeclarationIdentity:
    """Every source declaration's handle/``TypeDef`` adopts its own AST node
    id as its declaration identity. Reserved identities belong only to
    canonical fallback definitions used when no source declaration is loaded.
    """

    def test_the_first_declaration_in_a_program_is_not_read_as_having_no_identity(self) -> None:
        """A field-less declaration written as a program's very first item owns
        the lowest AST node id there is, and still carries a real declaration
        identity rather than the "no declaration identity" value."""
        checked = _check("record R\n  *\nlet r = R()\n()")
        record_def = next(
            item
            for item in checked.resolved.program.body.items
            if isinstance(item, RecordDef) and item.name == "R"
        )
        assert record_def.node_id == 0
        r = checked.type_env.get_type("R")
        assert isinstance(r, RecordType)
        assert r.decl_id != NO_DECL_ID
        typedef = checked.type_env.type_table.get(r.module_id, "R")
        assert typedef is not None
        assert typedef.decl_node_id != NO_DECL_ID

    def test_record_handle_and_typedef_adopt_the_declaration_node_id(self) -> None:
        checked = _check("record Point\n  x: int\n  y: int\nlet p = Point(x = 1, y = 2)\np")
        record_def = next(
            item
            for item in checked.resolved.program.body.items
            if isinstance(item, RecordDef) and item.name == "Point"
        )
        point = checked.type_env.get_type("Point")
        assert isinstance(point, RecordType)
        assert point.decl_id == record_def.node_id
        typedef = checked.type_env.type_table.get(point.module_id, "Point")
        assert typedef is not None
        assert typedef.decl_node_id == record_def.node_id

    def test_enum_handle_and_typedef_adopt_the_declaration_node_id(self) -> None:
        checked = _check("enum Color\n  | Red\n  | Green\n  | Blue\nlet c = Red\nc")
        enum_def = next(
            item
            for item in checked.resolved.program.body.items
            if isinstance(item, EnumDef) and item.name == "Color"
        )
        color = checked.type_env.get_type("Color")
        assert isinstance(color, EnumType)
        assert color.decl_id == enum_def.node_id
        typedef = checked.type_env.type_table.get(color.module_id, "Color")
        assert typedef is not None
        assert typedef.decl_node_id == enum_def.node_id

    def test_exception_handle_and_typedef_adopt_the_declaration_node_id(self) -> None:
        checked = _check(
            'exception Boom extends Exception\n  code: int\nBoom(code = 5, message = "m")'
        )
        exception_def = next(
            item
            for item in checked.resolved.program.body.items
            if isinstance(item, ExceptionDef) and item.name == "Boom"
        )
        boom = checked.type_env.get_type("Boom")
        assert isinstance(boom, ExceptionType)
        assert boom.decl_id == exception_def.node_id
        typedef = checked.type_env.type_table.get(boom.module_id, "Boom")
        assert typedef is not None
        assert typedef.decl_node_id == exception_def.node_id

    def test_generic_record_instantiation_preserves_declaration_identity_across_arguments(
        self,
    ) -> None:
        checked = _check(
            "record Box[T]\n  value: T\n"
            'let a: Box[int] = Box(value = 1)\nlet b: Box[text] = Box(value = "x")\n()'
        )
        a = _binding_value_type(checked, "a")
        b = _binding_value_type(checked, "b")
        assert isinstance(a, RecordType)
        assert isinstance(b, RecordType)
        assert a.decl_id != NO_DECL_ID
        assert a.decl_id == b.decl_id

    def test_generic_type_annotation_resolves_to_the_declarations_identity(self) -> None:
        """Applying a generic declaration's type arguments in an annotation
        instantiates its registered template, so the resolved handle names the
        declaration rather than losing its identity."""
        checked = _check("record Box[T]\n  value: T\ndef unwrap(b: Box[int]) -> int = b.value\n()")
        record_def = next(
            item
            for item in checked.resolved.program.body.items
            if isinstance(item, RecordDef) and item.name == "Box"
        )
        param_type = checked.function_signatures["unwrap"].params[0].type
        assert isinstance(param_type, RecordType)
        assert param_type.type_args == (IntType(),)
        assert param_type.decl_id == record_def.node_id

    def test_same_named_declarations_in_different_modules_have_different_identities(
        self, tmp_path: Path
    ) -> None:
        """Declaration identity distinguishes two declarations that share a
        bare name, which is what makes it an identity rather than a label."""
        checked = _check_program(
            tmp_path,
            {
                "mod_a": "record Point\n  x: int\n",
                "mod_b": "record Point\n  x: int\n",
                "entry": "import mod_a\nimport mod_b\n()\n",
            },
        )
        decl_ids = [
            handle.decl_id
            for mid, module in checked.modules.items()
            if not mid.is_entry and not mid.is_standard_library
            for handle in [module.type_env.get_type("Point")]
            if isinstance(handle, RecordType)
        ]
        assert len(decl_ids) == 2
        assert decl_ids[0] != decl_ids[1]
        assert all(decl_id != NO_DECL_ID for decl_id in decl_ids)

    def test_stdlib_declarations_of_reserved_names_adopt_their_source_identity(
        self, tmp_path: Path
    ) -> None:
        """Loading the standard library selects its declarations over the
        reserved fallbacks without making their identity depend on which
        standard-library module happens to declare the name."""
        checked = _check_program(tmp_path, {"entry": "()"})
        declared_reserved = set(RESERVED_NOMINAL_NAMES) - COMPATIBILITY_PRELUDE_TYPE_NAMES
        assert "Option" in declared_reserved
        for name in sorted(declared_reserved):
            declaring = [
                module
                for mid, module in checked.modules.items()
                if mid.is_standard_library
                and any(
                    isinstance(item, (RecordDef, EnumDef, ExceptionDef)) and item.name == name
                    for item in module.resolved.program.body.items
                )
            ]
            assert len(declaring) == 1, name
            module = declaring[0]
            declarations = {
                item.name: item
                for item in module.resolved.program.body.items
                if isinstance(item, (RecordDef, EnumDef, ExceptionDef))
            }
            handle = module.type_env.get_type(name)
            if handle is None:
                # Generic declarations register a template, not a bare handle.
                handle = module.type_env.all_generic_types()[name].template
            assert isinstance(handle, (RecordType, EnumType, ExceptionType))
            assert handle.decl_id == declarations[name].node_id, name
            assert handle.decl_id != reserved_nominal_id(name), name
            if isinstance(handle, EnumType):
                declaration = declarations[name]
                assert isinstance(declaration, EnumDef)
                source_members = {
                    member.name: member.node_id
                    for member in declaration.members
                    if isinstance(member, VariantDef)
                }
                resolved_members = module.type_env.type_table.enum_member_names(handle)
                assert {
                    member_name: member.decl_id for member_name, member in resolved_members.items()
                } == source_members
