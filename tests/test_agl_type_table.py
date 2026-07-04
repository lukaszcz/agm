"""Tests for the shared nominal type-declaration table.

Covers ``TypeDef``/``TypeTable`` in ``agm.agl.semantics.type_table`` — the
sole source of record/enum field and variant shapes, since ``RecordType``/
``EnumType`` handles carry none — and how it is populated: the type builder
(``typecheck/builder.py``), the graph pre-pass (``typecheck/graph.py``), and
REPL session accumulation (``TypeEnvironment.seed_from``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID, STD_CORE_ID, ModuleId
from agm.agl.parser import parse_program
from agm.agl.repl import ReplSession
from agm.agl.scope import resolve
from agm.agl.scope.graph import resolve_graph
from agm.agl.semantics.type_table import (
    BUILTIN_PRELUDE_TYPE_DEFS,
    TypeDef,
    TypeTable,
    comparable_types,
    create_seeded_type_table,
)
from agm.agl.semantics.types import (
    BUILTIN_PRELUDE_TYPES,
    AgentType,
    DictType,
    EnumType,
    ExceptionType,
    FunctionType,
    IntType,
    ListType,
    RecordType,
    TextType,
    TypeVarType,
    UnitType,
)
from agm.agl.syntax.nodes import LetDecl, ParamKind, VarDecl
from agm.agl.typecheck import CheckedProgram, check
from agm.agl.typecheck.graph import check_graph
from tests.agl.ir_harness import make_graph_from_files

_CAPS = HostCapabilities(
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

_LIB_ID = ModuleId.from_dotted("lib")


def _check(src: str) -> CheckedProgram:
    """Parse + resolve + check a single-module AgL program."""
    return check(resolve(parse_program(src)), _CAPS)


def _check_graph(tmp_path: Path, modules: dict[str, str]):
    """Build and typecheck a multi-module graph; returns CheckedModuleGraph."""
    graph = make_graph_from_files(tmp_path, modules)
    rgraph = resolve_graph(graph)
    return check_graph(rgraph, _CAPS)


def _binding_value_type(checked: CheckedProgram, name: str):
    """Inferred type of the RHS of the top-level ``let``/``var <name> = ...``."""
    for item in checked.resolved.program.body.items:
        if isinstance(item, (LetDecl, VarDecl)) and item.name == name:
            return checked.node_types[item.value.node_id]
    raise AssertionError(f"no top-level binding named {name!r}")


# ---------------------------------------------------------------------------
# register / get
# ---------------------------------------------------------------------------


class TestTypeDefHandle:
    def test_record_handle(self) -> None:
        typedef = TypeDef(kind="record", name="Point", module_id=ENTRY_ID)
        assert typedef.handle() == RecordType(name="Point", module_id=ENTRY_ID)

    def test_enum_handle(self) -> None:
        typedef = TypeDef(kind="enum", name="Color", module_id=ENTRY_ID)
        assert typedef.handle() == EnumType(name="Color", module_id=ENTRY_ID)

    def test_generic_record_handle_with_type_args(self) -> None:
        typedef = TypeDef(kind="record", name="Box", module_id=ENTRY_ID, type_params=("T",))
        handle = typedef.handle(type_args=(IntType(),))
        assert handle == RecordType(name="Box", type_args=(IntType(),), module_id=ENTRY_ID)

    def test_exception_kind_unsupported(self) -> None:
        typedef = TypeDef(kind="exception", name="Boom", module_id=ENTRY_ID)
        with pytest.raises(ValueError, match="does not support kind"):
            typedef.handle()


class TestRegisterAndGet:
    def test_register_and_get_record_in_entry_module(self) -> None:
        table = TypeTable()
        typedef = TypeDef(
            kind="record",
            name="Point",
            module_id=ENTRY_ID,
            fields=(("x", IntType()), ("y", IntType())),
        )
        table.register(typedef)
        assert table.get(ENTRY_ID, "Point") == typedef

    def test_register_and_get_enum_in_non_entry_module(self) -> None:
        table = TypeTable()
        typedef = TypeDef(
            kind="enum",
            name="Color",
            module_id=_LIB_ID,
            variants=(("Red", ()), ("Blue", ())),
        )
        table.register(typedef)
        assert table.get(_LIB_ID, "Color") == typedef

    def test_get_missing_key_returns_none(self) -> None:
        table = TypeTable()
        assert table.get(ENTRY_ID, "Nope") is None

    def test_same_name_in_different_modules_are_independent_entries(self) -> None:
        table = TypeTable()
        entry_def = TypeDef(
            kind="record", name="Widget", module_id=ENTRY_ID, fields=(("a", IntType()),)
        )
        lib_def = TypeDef(
            kind="record", name="Widget", module_id=_LIB_ID, fields=(("b", TextType()),)
        )
        table.register(entry_def)
        table.register(lib_def)
        assert table.get(ENTRY_ID, "Widget") == entry_def
        assert table.get(_LIB_ID, "Widget") == lib_def


# ---------------------------------------------------------------------------
# record_fields / enum_variants on non-generic handles
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
            )
        )
        handle = RecordType(name="Point", module_id=ENTRY_ID)
        assert dict(table.record_fields(handle)) == {"x": IntType(), "y": IntType()}

    def test_enum_variants_non_generic(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="enum",
                name="Color",
                module_id=ENTRY_ID,
                variants=(("Red", ()), ("Custom", (("hex", TextType()),))),
            )
        )
        handle = EnumType(name="Color", module_id=ENTRY_ID)
        result = table.enum_variants(handle)
        assert {v: dict(f) for v, f in result.items()} == {
            "Red": {},
            "Custom": {"hex": TextType()},
        }

    def test_record_fields_missing_def_raises_keyerror(self) -> None:
        table = TypeTable()
        handle = RecordType(name="Ghost", module_id=ENTRY_ID)
        with pytest.raises(KeyError):
            table.record_fields(handle)

    def test_enum_variants_missing_def_raises_keyerror(self) -> None:
        table = TypeTable()
        handle = EnumType(name="Ghost", module_id=ENTRY_ID)
        with pytest.raises(KeyError):
            table.enum_variants(handle)

    def test_record_fields_raises_when_key_registered_as_enum(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(kind="enum", name="Color", module_id=ENTRY_ID, variants=(("Red", ()),))
        )
        handle = RecordType(name="Color", module_id=ENTRY_ID)
        with pytest.raises(AssertionError):
            table.record_fields(handle)

    def test_enum_variants_raises_when_key_registered_as_record(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(kind="record", name="Point", module_id=ENTRY_ID, fields=(("x", IntType()),))
        )
        handle = EnumType(name="Point", module_id=ENTRY_ID)
        with pytest.raises(AssertionError):
            table.enum_variants(handle)


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
                fields=(("message", TextType()), ("trace_id", TextType())),
                abstract=True,
            )
        )
        handle = ExceptionType(name="Exception", module_id=ENTRY_ID)
        assert dict(table.exception_fields(handle)) == {
            "message": TextType(),
            "trace_id": TextType(),
        }

    def test_exception_fields_flattens_base_chain_root_mid_leaf_order(self) -> None:
        """A three-level ``extends`` chain flattens base-first, in declaration order."""
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="Root",
                module_id=ENTRY_ID,
                fields=(("message", TextType()), ("trace_id", TextType())),
                abstract=True,
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Mid",
                module_id=ENTRY_ID,
                fields=(("code", IntType()),),
                base=(ENTRY_ID, "Root"),
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Leaf",
                module_id=ENTRY_ID,
                fields=(("detail", TextType()),),
                base=(ENTRY_ID, "Mid"),
            )
        )
        handle = ExceptionType(name="Leaf", module_id=ENTRY_ID)
        assert list(table.exception_fields(handle).keys()) == [
            "message",
            "trace_id",
            "code",
            "detail",
        ]

    def test_exception_fields_resolves_cross_module_base(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="Base",
                module_id=_LIB_ID,
                fields=(("message", TextType()), ("trace_id", TextType())),
                abstract=True,
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Child",
                module_id=ENTRY_ID,
                fields=(("code", IntType()),),
                base=(_LIB_ID, "Base"),
            )
        )
        handle = ExceptionType(name="Child", module_id=ENTRY_ID)
        assert dict(table.exception_fields(handle)) == {
            "message": TextType(),
            "trace_id": TextType(),
            "code": IntType(),
        }

    def test_exception_fields_returns_same_object_for_same_handle(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception", name="Boom", module_id=ENTRY_ID, fields=(("code", IntType()),)
            )
        )
        handle = ExceptionType(name="Boom", module_id=ENTRY_ID)
        first = table.exception_fields(handle)
        second = table.exception_fields(handle)
        assert first is second

    def test_exception_fields_missing_def_raises_keyerror(self) -> None:
        table = TypeTable()
        handle = ExceptionType(name="Ghost", module_id=ENTRY_ID)
        with pytest.raises(KeyError):
            table.exception_fields(handle)

    def test_exception_fields_raises_when_key_registered_as_record(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(kind="record", name="Point", module_id=ENTRY_ID, fields=(("x", IntType()),))
        )
        handle = ExceptionType(name="Point", module_id=ENTRY_ID)
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
                base=(ENTRY_ID, "B"),
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="B",
                module_id=ENTRY_ID,
                base=(ENTRY_ID, "A"),
            )
        )
        handle = ExceptionType(name="A", module_id=ENTRY_ID)
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
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Child",
                module_id=ENTRY_ID,
                fields=(("code", IntType()),),
                base=(ENTRY_ID, "Root"),
            )
        )
        root_def = table.exception_def(ExceptionType(name="Root", module_id=ENTRY_ID))
        assert root_def.abstract is True
        assert root_def.base is None
        child_def = table.exception_def(ExceptionType(name="Child", module_id=ENTRY_ID))
        assert child_def.abstract is False
        assert child_def.base == (ENTRY_ID, "Root")

    def test_exception_def_missing_def_raises_keyerror(self) -> None:
        table = TypeTable()
        handle = ExceptionType(name="Ghost", module_id=ENTRY_ID)
        with pytest.raises(KeyError):
            table.exception_def(handle)

    def test_exception_def_raises_when_key_registered_as_enum(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(kind="enum", name="Color", module_id=ENTRY_ID, variants=(("Red", ()),))
        )
        handle = ExceptionType(name="Color", module_id=ENTRY_ID)
        with pytest.raises(AssertionError):
            table.exception_def(handle)


# ---------------------------------------------------------------------------
# TypeTable.exception_field_kinds — own fields honor their declared kind;
# only the extends-chain flattening order (base-first) is exception-specific.
# ---------------------------------------------------------------------------


class TestExceptionFieldKinds:
    """``exception_field_kinds`` returns ``ParamKind.value`` strings (not the
    enum): ``semantics`` may not import ``syntax.nodes``, so ``TypeDef.
    field_kinds`` stores the stable string values instead (converted back to
    ``ParamKind`` by ``typecheck.env``)."""

    def test_root_only_excludes_trace_id(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="Exception",
                module_id=ENTRY_ID,
                fields=(("message", TextType()), ("trace_id", TextType())),
                abstract=True,
                field_kinds=(ParamKind.NAMED_ONLY.value, ParamKind.NAMED_ONLY.value),
            )
        )
        handle = ExceptionType(name="Exception", module_id=ENTRY_ID)
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
                fields=(("message", TextType()), ("trace_id", TextType())),
                abstract=True,
                field_kinds=(ParamKind.NAMED_ONLY.value, ParamKind.NAMED_ONLY.value),
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Mid",
                module_id=ENTRY_ID,
                fields=(("code", IntType()),),
                base=(ENTRY_ID, "Root"),
                field_kinds=(ParamKind.STANDARD.value,),
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Leaf",
                module_id=ENTRY_ID,
                fields=(("detail", TextType()),),
                base=(ENTRY_ID, "Mid"),
                field_kinds=(ParamKind.POSITIONAL_ONLY.value,),
            )
        )
        handle = ExceptionType(name="Leaf", module_id=ENTRY_ID)
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
                fields=(("message", TextType()), ("trace_id", TextType())),
                abstract=True,
                field_kinds=(ParamKind.NAMED_ONLY.value, ParamKind.NAMED_ONLY.value),
            )
        )
        table.register(
            TypeDef(
                kind="exception",
                name="Child",
                module_id=ENTRY_ID,
                fields=(("code", IntType()),),
                base=(_LIB_ID, "Base"),
                field_kinds=(ParamKind.STANDARD.value,),
            )
        )
        handle = ExceptionType(name="Child", module_id=ENTRY_ID)
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
            )
        )
        handle = ExceptionType(name="Boom", module_id=ENTRY_ID)
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
            TypeDef(kind="record", name="Point", module_id=ENTRY_ID, fields=(("x", IntType()),))
        )
        handle = ExceptionType(name="Point", module_id=ENTRY_ID)
        with pytest.raises(AssertionError):
            table.exception_field_kinds(handle)

    def test_raises_on_cyclic_base_chain(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(kind="exception", name="A", module_id=ENTRY_ID, base=(ENTRY_ID, "B"))
        )
        table.register(
            TypeDef(kind="exception", name="B", module_id=ENTRY_ID, base=(ENTRY_ID, "A"))
        )
        handle = ExceptionType(name="A", module_id=ENTRY_ID)
        with pytest.raises(AssertionError, match="cyclic exception base chain"):
            table.exception_field_kinds(handle)

    def test_unregister_invalidates_cached_field_kinds(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="Boom",
                module_id=ENTRY_ID,
                fields=(("code", IntType()),),
                field_kinds=(ParamKind.NAMED_ONLY.value,),
            )
        )
        handle = ExceptionType(name="Boom", module_id=ENTRY_ID)
        assert table.exception_field_kinds(handle) == (("code", ParamKind.NAMED_ONLY.value),)

        table.unregister(ENTRY_ID, "Boom")
        table.register(
            TypeDef(
                kind="exception",
                name="Boom",
                module_id=ENTRY_ID,
                fields=(("code", IntType()),),
                field_kinds=(ParamKind.STANDARD.value,),
            )
        )
        assert table.exception_field_kinds(handle) == (("code", ParamKind.STANDARD.value),)


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
                    ("firsts", ListType(TypeVarType("T"))),
                    ("seconds", DictType(TypeVarType("U"))),
                ),
            )
        )
        handle = RecordType(
            name="Pair", type_args=(IntType(), TextType()), module_id=ENTRY_ID
        )
        result = table.record_fields(handle)
        assert dict(result) == {
            "first": IntType(),
            "second": TextType(),
            "firsts": ListType(IntType()),
            "seconds": DictType(TextType()),
        }

    def test_enum_variants_substitutes_type_args(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="enum",
                name="Maybe",
                module_id=ENTRY_ID,
                type_params=("T",),
                variants=(("None", ()), ("Just", (("value", TypeVarType("T")),))),
            )
        )
        handle = EnumType(
            name="Maybe", type_args=(IntType(),), module_id=ENTRY_ID
        )
        result = table.enum_variants(handle)
        assert {v: dict(f) for v, f in result.items()} == {
            "None": {},
            "Just": {"value": IntType()},
        }


# ---------------------------------------------------------------------------
# Memoization
# ---------------------------------------------------------------------------


class TestMemoization:
    def test_record_fields_returns_same_object_for_same_handle(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(kind="record", name="Point", module_id=ENTRY_ID, fields=(("x", IntType()),))
        )
        handle = RecordType(name="Point", module_id=ENTRY_ID)
        first = table.record_fields(handle)
        second = table.record_fields(handle)
        assert first is second

    def test_enum_variants_returns_same_object_for_same_handle(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(kind="enum", name="Color", module_id=ENTRY_ID, variants=(("Red", ()),))
        )
        handle = EnumType(name="Color", module_id=ENTRY_ID)
        first = table.enum_variants(handle)
        second = table.enum_variants(handle)
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
            )
        )
        int_handle = RecordType(
            name="Box", type_args=(IntType(),), module_id=ENTRY_ID
        )
        text_handle = RecordType(
            name="Box", type_args=(TextType(),), module_id=ENTRY_ID
        )
        assert dict(table.record_fields(int_handle)) == {"value": IntType()}
        assert dict(table.record_fields(text_handle)) == {"value": TextType()}
        # Re-fetching the first handle still returns its own cached result.
        assert dict(table.record_fields(int_handle)) == {"value": IntType()}

    def test_enum_variants_caches_each_generic_instantiation_separately(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="enum",
                name="Maybe",
                module_id=ENTRY_ID,
                type_params=("T",),
                variants=(("Just", (("value", TypeVarType("T")),)),),
            )
        )
        int_handle = EnumType(
            name="Maybe", type_args=(IntType(),), module_id=ENTRY_ID
        )
        text_handle = EnumType(
            name="Maybe", type_args=(TextType(),), module_id=ENTRY_ID
        )
        assert {v: dict(f) for v, f in table.enum_variants(int_handle).items()} == {
            "Just": {"value": IntType()}
        }
        assert {v: dict(f) for v, f in table.enum_variants(text_handle).items()} == {
            "Just": {"value": TextType()}
        }


# ---------------------------------------------------------------------------
# Re-registration semantics
# ---------------------------------------------------------------------------


class TestReRegistration:
    def test_identical_re_registration_is_a_no_op(self) -> None:
        table = TypeTable()
        typedef = TypeDef(
            kind="record", name="Point", module_id=ENTRY_ID, fields=(("x", IntType()),)
        )
        table.register(typedef)
        table.register(
            TypeDef(kind="record", name="Point", module_id=ENTRY_ID, fields=(("x", IntType()),))
        )
        assert table.get(ENTRY_ID, "Point") == typedef

    def test_conflicting_re_registration_raises(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(kind="record", name="Point", module_id=ENTRY_ID, fields=(("x", IntType()),))
        )
        with pytest.raises(AssertionError):
            table.register(
                TypeDef(
                    kind="record", name="Point", module_id=ENTRY_ID, fields=(("x", TextType()),)
                )
            )


# ---------------------------------------------------------------------------
# unregister()
# ---------------------------------------------------------------------------


class TestUnregister:
    def test_unregister_removes_the_def(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(kind="record", name="Point", module_id=ENTRY_ID, fields=(("x", IntType()),))
        )
        table.unregister(ENTRY_ID, "Point")
        assert table.get(ENTRY_ID, "Point") is None

    def test_unregister_missing_key_is_a_no_op(self) -> None:
        table = TypeTable()
        table.unregister(ENTRY_ID, "Nope")  # must not raise
        assert table.get(ENTRY_ID, "Nope") is None

    def test_unregister_then_register_allows_a_different_shape(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(kind="record", name="Point", module_id=ENTRY_ID, fields=(("x", IntType()),))
        )
        table.unregister(ENTRY_ID, "Point")
        new_def = TypeDef(
            kind="record", name="Point", module_id=ENTRY_ID, fields=(("x", TextType()),)
        )
        table.register(new_def)  # would raise if the old entry were still present
        assert table.get(ENTRY_ID, "Point") == new_def

    def test_unregister_invalidates_cached_substitution(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(kind="record", name="Point", module_id=ENTRY_ID, fields=(("x", IntType()),))
        )
        handle = RecordType(name="Point", module_id=ENTRY_ID)
        assert dict(table.record_fields(handle)) == {"x": IntType()}

        table.unregister(ENTRY_ID, "Point")
        table.register(
            TypeDef(kind="record", name="Point", module_id=ENTRY_ID, fields=(("x", TextType()),))
        )
        assert dict(table.record_fields(handle)) == {"x": TextType()}

    def test_unregister_invalidates_cached_enum_substitution(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(kind="enum", name="Color", module_id=ENTRY_ID, variants=(("Red", ()),))
        )
        handle = EnumType(name="Color", module_id=ENTRY_ID)
        assert {v: dict(f) for v, f in table.enum_variants(handle).items()} == {"Red": {}}

        table.unregister(ENTRY_ID, "Color")
        table.register(
            TypeDef(
                kind="enum",
                name="Color",
                module_id=ENTRY_ID,
                variants=(("Red", ()), ("Blue", ())),
            )
        )
        assert {v: dict(f) for v, f in table.enum_variants(handle).items()} == {
            "Red": {},
            "Blue": {},
        }


# ---------------------------------------------------------------------------
# entries() / merge_from()
# ---------------------------------------------------------------------------


class TestEntriesAndMerge:
    def test_entries_returns_all_registered_defs(self) -> None:
        table = TypeTable()
        a = TypeDef(kind="record", name="A", module_id=ENTRY_ID, fields=())
        b = TypeDef(kind="record", name="B", module_id=ENTRY_ID, fields=())
        table.register(a)
        table.register(b)
        assert set(table.entries()) == {a, b}

    def test_merge_from_copies_new_entries_and_skips_identical(self) -> None:
        source = TypeTable()
        shared = TypeDef(kind="record", name="Shared", module_id=ENTRY_ID, fields=())
        new = TypeDef(kind="record", name="New", module_id=ENTRY_ID, fields=())
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
        new_def = TypeDef(kind="record", name="X", module_id=ENTRY_ID, fields=(("a", IntType()),))
        source.register(new_def)

        target = TypeTable()
        target.register(
            TypeDef(kind="record", name="X", module_id=ENTRY_ID, fields=(("a", TextType()),))
        )

        target.merge_from(source)

        assert target.get(ENTRY_ID, "X") == new_def

    def test_merge_from_keeps_cached_substitution_when_entry_is_unchanged(self) -> None:
        # An identical incoming entry must not perturb an already-cached
        # substitution: the same handle keeps resolving to the same object.
        shared = TypeDef(
            kind="record", name="Shared", module_id=ENTRY_ID, fields=(("a", IntType()),)
        )
        source = TypeTable()
        source.register(shared)

        target = TypeTable()
        target.register(shared)
        handle = RecordType(name="Shared", module_id=ENTRY_ID)
        before = target.record_fields(handle)

        target.merge_from(source)

        after = target.record_fields(handle)
        assert after is before

    def test_merge_from_invalidates_stale_cached_substitution(self) -> None:
        source = TypeTable()
        new_def = TypeDef(kind="record", name="X", module_id=ENTRY_ID, fields=(("a", IntType()),))
        source.register(new_def)

        target = TypeTable()
        target.register(
            TypeDef(kind="record", name="X", module_id=ENTRY_ID, fields=(("a", TextType()),))
        )
        handle = RecordType(name="X", module_id=ENTRY_ID)
        assert dict(target.record_fields(handle)) == {"a": TextType()}

        target.merge_from(source)

        assert dict(target.record_fields(handle)) == {"a": IntType()}


# ---------------------------------------------------------------------------
# Built-in prelude seeding
# ---------------------------------------------------------------------------


class TestBuiltinSeeding:
    def test_all_prelude_types_resolvable(self) -> None:
        table = create_seeded_type_table()
        for name, typ in BUILTIN_PRELUDE_TYPES.items():
            typedef = table.get(PRELUDE_ID, name)
            assert typedef is not None
            expected = BUILTIN_PRELUDE_TYPE_DEFS[name]
            if isinstance(typ, RecordType):
                handle = RecordType(name=name, module_id=PRELUDE_ID)
                assert dict(table.record_fields(handle)) == dict(expected.fields)
            else:
                handle = EnumType(name=name, module_id=PRELUDE_ID)
                result = table.enum_variants(handle)
                assert {v: dict(f) for v, f in result.items()} == {
                    vname: dict(vfields) for vname, vfields in expected.variants
                }

    def test_generic_option_seeded_under_std_core(self) -> None:
        table = create_seeded_type_table()
        typedef = table.get(STD_CORE_ID, "Option")
        assert typedef is not None
        assert typedef.type_params == ("T",)
        handle = EnumType(
            name="Option", type_args=(TextType(),), module_id=STD_CORE_ID
        )
        result = table.enum_variants(handle)
        assert {v: dict(f) for v, f in result.items()} == {
            "None": {},
            "Some": {"value": TextType()},
        }


# ---------------------------------------------------------------------------
# Every env-registered type has a matching table def: single-module
# ---------------------------------------------------------------------------


class TestEnvTypeHasMatchingTableDefSingleModule:
    """Every type registered in the env has a matching ``TypeDef`` in the
    shared table (the only place shapes live now that handles carry none)."""

    def test_non_generic_record(self) -> None:
        checked = _check(
            "record Point\n  x: int\n  y: int\nlet p = Point(x = 1, y = 2)\np"
        )
        point = checked.type_env.get_type("Point")
        assert isinstance(point, RecordType)
        table = checked.type_env.type_table
        typedef = table.get(point.module_id, "Point")
        assert typedef is not None
        assert typedef.kind == "record"
        assert dict(table.record_fields(point)) == {"x": IntType(), "y": IntType()}

    def test_generic_record(self) -> None:
        checked = _check(
            "record Box[T]\n  value: T\nlet b: Box[int] = Box(value = 1)\nb"
        )
        box = _binding_value_type(checked, "b")
        assert isinstance(box, RecordType)
        table = checked.type_env.type_table
        typedef = table.get(box.module_id, "Box")
        assert typedef is not None
        assert typedef.type_params == ("T",)
        assert dict(table.record_fields(box)) == {"value": IntType()}

    def test_non_generic_enum(self) -> None:
        checked = _check(
            "enum Color\n  | Red\n  | Green\n  | Blue\nlet c = Red\nc"
        )
        color = checked.type_env.get_type("Color")
        assert isinstance(color, EnumType)
        table = checked.type_env.type_table
        typedef = table.get(color.module_id, "Color")
        assert typedef is not None
        assert typedef.kind == "enum"
        result = table.enum_variants(color)
        assert {v: dict(f) for v, f in result.items()} == {
            "Red": {},
            "Green": {},
            "Blue": {},
        }

    def test_generic_enum(self) -> None:
        checked = _check(
            "enum Maybe[T]\n  | none\n  | just(value: T)\nlet m = just(value = 1)\nm"
        )
        maybe = _binding_value_type(checked, "m")
        assert isinstance(maybe, EnumType)
        table = checked.type_env.type_table
        typedef = table.get(maybe.module_id, "Maybe")
        assert typedef is not None
        assert typedef.type_params == ("T",)
        result = table.enum_variants(maybe)
        assert {v: dict(f) for v, f in result.items()} == {
            "none": {},
            "just": {"value": IntType()},
        }


# ---------------------------------------------------------------------------
# Every env-registered type has a matching table def: graph mode
# ---------------------------------------------------------------------------


class TestEnvTypeHasMatchingTableDefGraphMode:
    def test_record_declared_in_one_module_reachable_from_another(
        self, tmp_path: Path
    ) -> None:
        modules = {
            "entry": (
                "import mylib\n"
                "def make() -> mylib::Point = mylib::makePoint()\n"
                "let p = make()\n"
                "p"
            ),
            "mylib": (
                "record Point\n  x: int\n  y: int\n"
                "def makePoint() -> Point = Point(x = 1, y = 2)"
            ),
        }
        cg = _check_graph(tmp_path, modules)
        mylib_id = ModuleId.from_dotted("mylib")
        point = cg.graph_type_table[(mylib_id, "Point")]
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
# comparable_types / _has_no_value_equality: table-aware record/enum walk
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
            )
        )
        handle = RecordType(name="Point", module_id=ENTRY_ID)
        assert comparable_types(handle, handle, table) is True

    def test_generic_record_agent_field_via_instantiation_not_comparable(self) -> None:
        # A generic record's field template is a bare type variable; only once
        # a concrete handle instantiates it with an agent type does the field
        # actually carry a no-equality type — record_fields substitutes
        # type_args into the template to expose this.
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Box",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(("value", TypeVarType("T")),),
            )
        )
        agent_handle = RecordType(
            name="Box", type_args=(AgentType(),), module_id=ENTRY_ID
        )
        assert comparable_types(agent_handle, agent_handle, table) is False

        int_handle = RecordType(
            name="Box", type_args=(IntType(),), module_id=ENTRY_ID
        )
        assert comparable_types(int_handle, int_handle, table) is True

    def test_generic_enum_function_variant_via_instantiation_not_comparable(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="enum",
                name="Holder",
                module_id=ENTRY_ID,
                type_params=("T",),
                variants=(("None", ()), ("Some", (("value", TypeVarType("T")),))),
            )
        )
        fn_type = FunctionType(params=(IntType(),), result=IntType())
        fn_handle = EnumType(
            name="Holder", type_args=(fn_type,), module_id=ENTRY_ID
        )
        assert comparable_types(fn_handle, fn_handle, table) is False

        text_handle = EnumType(
            name="Holder", type_args=(TextType(),), module_id=ENTRY_ID
        )
        assert comparable_types(text_handle, text_handle, table) is True

    def test_record_with_unit_nested_in_list_field_not_comparable(self) -> None:
        # Nested depth: the record field itself is a list, whose element type
        # is the type parameter — instantiating with unit makes the list of
        # unit values transitively non-comparable.
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Wrapper",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(("items", ListType(TypeVarType("T"))),),
            )
        )
        handle = RecordType(
            name="Wrapper", type_args=(UnitType(),), module_id=ENTRY_ID
        )
        assert comparable_types(handle, handle, table) is False

    def test_enum_variant_with_agent_nested_in_dict_field_not_comparable(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="enum",
                name="Bag",
                module_id=ENTRY_ID,
                type_params=("T",),
                variants=(("Full", (("byKey", DictType(TypeVarType("T"))),)),),
            )
        )
        handle = EnumType(
            name="Bag", type_args=(AgentType(),), module_id=ENTRY_ID
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
            )
        )
        exc = ExceptionType(name="Failure", module_id=ENTRY_ID)
        assert comparable_types(exc, exc, table) is False

    def test_exception_with_only_scalar_fields_comparable(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception", name="Failure", module_id=ENTRY_ID, fields=(("code", IntType()),)
            )
        )
        exc = ExceptionType(name="Failure", module_id=ENTRY_ID)
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
            )
        )
        exc = ExceptionType(name="Failure", module_id=ENTRY_ID)
        table.register(
            TypeDef(kind="record", name="Report", module_id=ENTRY_ID, fields=(("cause", exc),))
        )
        handle = RecordType(name="Report", module_id=ENTRY_ID)
        assert comparable_types(handle, handle, table) is False
