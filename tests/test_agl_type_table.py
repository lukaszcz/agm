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
from agm.agl.semantics.analyses import compute_finite_closure
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
        "json": frozenset({"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}),
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
        handle = RecordType(name="Pair", type_args=(IntType(), TextType()), module_id=ENTRY_ID)
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
        handle = EnumType(name="Maybe", type_args=(IntType(),), module_id=ENTRY_ID)
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
        int_handle = RecordType(name="Box", type_args=(IntType(),), module_id=ENTRY_ID)
        text_handle = RecordType(name="Box", type_args=(TextType(),), module_id=ENTRY_ID)
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
        int_handle = EnumType(name="Maybe", type_args=(IntType(),), module_id=ENTRY_ID)
        text_handle = EnumType(name="Maybe", type_args=(TextType(),), module_id=ENTRY_ID)
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
        handle = EnumType(name="Option", type_args=(TextType(),), module_id=STD_CORE_ID)
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
        result = table.enum_variants(color)
        assert {v: dict(f) for v, f in result.items()} == {
            "Red": {},
            "Green": {},
            "Blue": {},
        }

    def test_generic_enum(self) -> None:
        checked = _check("enum Maybe[T]\n  | none\n  | just(value: T)\nlet m = just(value = 1)\nm")
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
    def test_record_declared_in_one_module_reachable_from_another(self, tmp_path: Path) -> None:
        modules = {
            "entry": (
                "import mylib\ndef make() -> mylib::Point = mylib::makePoint()\nlet p = make()\np"
            ),
            "mylib": (
                "record Point\n  x: int\n  y: int\ndef makePoint() -> Point = Point(x = 1, y = 2)"
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
        agent_handle = RecordType(name="Box", type_args=(AgentType(),), module_id=ENTRY_ID)
        assert comparable_types(agent_handle, agent_handle, table) is False

        int_handle = RecordType(name="Box", type_args=(IntType(),), module_id=ENTRY_ID)
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
        fn_handle = EnumType(name="Holder", type_args=(fn_type,), module_id=ENTRY_ID)
        assert comparable_types(fn_handle, fn_handle, table) is False

        text_handle = EnumType(name="Holder", type_args=(TextType(),), module_id=ENTRY_ID)
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
        handle = RecordType(name="Wrapper", type_args=(UnitType(),), module_id=ENTRY_ID)
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
        handle = EnumType(name="Bag", type_args=(AgentType(),), module_id=ENTRY_ID)
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

    def test_record_referencing_already_flagged_record_not_comparable(self) -> None:
        # X is unconditionally non-comparable (a function field); Y's only
        # field is a bare reference to X (not through an exception, unlike
        # the test above) — Y must inherit X's flag via the already-computed
        # fixpoint fact, not by re-walking X's own fields.
        table = TypeTable()
        fn_type = FunctionType(params=(), result=IntType())
        table.register(
            TypeDef(kind="record", name="X", module_id=ENTRY_ID, fields=(("fn", fn_type),))
        )
        table.register(
            TypeDef(
                kind="record",
                name="Y",
                module_id=ENTRY_ID,
                fields=(("x", RecordType(name="X", module_id=ENTRY_ID)),),
            )
        )
        handle = RecordType(name="Y", module_id=ENTRY_ID)
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
            )
        )
        handle = RecordType(name="Y", module_id=ENTRY_ID)
        assert comparable_types(handle, handle, table) is True

    def test_unregistered_handle_defaults_to_comparable(self) -> None:
        # Querying comparability of a handle whose own declaration was never
        # registered at all (as opposed to one merely referenced by a field)
        # is likewise defensive rather than a crash.
        table = TypeTable()
        handle = RecordType(name="Ghost", module_id=ENTRY_ID)
        assert comparable_types(handle, handle, table) is True

    def test_recursive_tree_is_comparable(self) -> None:
        # A self-referential enum (list/dict guard not even needed for
        # equality — only for inhabitation): the equality-capability fixpoint
        # must terminate on a cycle instead of recursing through the same
        # declaration's fields forever.
        table = TypeTable()
        table.register(
            TypeDef(
                kind="enum",
                name="Tree",
                module_id=ENTRY_ID,
                variants=(
                    ("Leaf", ()),
                    (
                        "Node",
                        (
                            ("value", IntType()),
                            ("left", EnumType(name="Tree", module_id=ENTRY_ID)),
                            ("right", EnumType(name="Tree", module_id=ENTRY_ID)),
                        ),
                    ),
                ),
            )
        )
        handle = EnumType(name="Tree", module_id=ENTRY_ID)
        assert comparable_types(handle, handle, table) is True

    def test_recursive_type_with_function_field_at_depth_not_comparable(self) -> None:
        # Same recursive shape as above, but one variant carries a function
        # field: the whole recursive type is non-comparable, exactly as a
        # non-recursive type containing a function field would be.
        table = TypeTable()
        handler_type = FunctionType(params=(), result=IntType())
        table.register(
            TypeDef(
                kind="enum",
                name="Tree",
                module_id=ENTRY_ID,
                variants=(
                    ("Leaf", ()),
                    ("Handler", (("fn", handler_type),)),
                    (
                        "Node",
                        (
                            ("left", EnumType(name="Tree", module_id=ENTRY_ID)),
                            ("right", EnumType(name="Tree", module_id=ENTRY_ID)),
                        ),
                    ),
                ),
            )
        )
        handle = EnumType(name="Tree", module_id=ENTRY_ID)
        assert comparable_types(handle, handle, table) is False

    def test_mutually_recursive_records_are_comparable(self) -> None:
        # A/B are mutually recursive through a list guard (inhabited) and
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
                    ("bs", ListType(RecordType(name="B", module_id=ENTRY_ID))),
                ),
            )
        )
        table.register(
            TypeDef(
                kind="record",
                name="B",
                module_id=ENTRY_ID,
                fields=(("a", RecordType(name="A", module_id=ENTRY_ID)),),
            )
        )
        a_handle = RecordType(name="A", module_id=ENTRY_ID)
        b_handle = RecordType(name="B", module_id=ENTRY_ID)
        assert comparable_types(a_handle, a_handle, table) is True
        assert comparable_types(b_handle, b_handle, table) is True


# ---------------------------------------------------------------------------
# Finiteness (instantiation-closure) analysis
# ---------------------------------------------------------------------------


def _pair_def(name: str = "Pair") -> TypeDef:
    """A plain non-recursive generic record with two independent parameters."""
    return TypeDef(
        kind="record",
        name=name,
        module_id=ENTRY_ID,
        type_params=("X", "Y"),
        fields=(("x", TypeVarType("X")), ("y", TypeVarType("Y"))),
    )


class TestFiniteClosure:
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
                        ListType(
                            RecordType("Tree", type_args=(TypeVarType("T"),), module_id=ENTRY_ID)
                        ),
                    ),
                ),
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
                    ("constref", RecordType("R", type_args=(IntType(),), module_id=ENTRY_ID)),
                ),
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "R") is True

    def test_permutation_cycle_is_finite(self) -> None:
        # Swap[A, B] referencing Swap[B, A]: each parameter is passed through
        # to a DIFFERENT slot unchanged (never a proper subterm), so the
        # A -> B -> A parameter cycle has no growing edge.
        table = TypeTable()
        table.register(
            TypeDef(
                kind="enum",
                name="Swap",
                module_id=ENTRY_ID,
                type_params=("A", "B"),
                variants=(
                    ("Base", (("a", TypeVarType("A")), ("b", TypeVarType("B")))),
                    (
                        "Rec",
                        (
                            (
                                "inner",
                                EnumType(
                                    "Swap",
                                    type_args=(TypeVarType("B"), TypeVarType("A")),
                                    module_id=ENTRY_ID,
                                ),
                            ),
                        ),
                    ),
                ),
            )
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
                                ),
                            ),
                            module_id=ENTRY_ID,
                        ),
                    ),
                ),
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "Perfect") is False
        # Pair itself is unrelated (non-recursive) and stays finite.
        assert table.has_finite_closure(ENTRY_ID, "Pair") is True

    def test_growing_via_list_is_infinite(self) -> None:
        # P[T] referencing P[list[T]]: T occurs under the list constructor,
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
                            "P", type_args=(ListType(TypeVarType("T")),), module_id=ENTRY_ID
                        ),
                    ),
                ),
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
                                ),
                            ),
                            module_id=ENTRY_ID,
                        ),
                    ),
                ),
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
                    ("a", RecordType("A", type_args=(TypeVarType("T"),), module_id=ENTRY_ID)),
                ),
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
                    ("d", RecordType("D", type_args=(TypeVarType("T"),), module_id=ENTRY_ID)),
                ),
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
                                ),
                            ),
                            module_id=ENTRY_ID,
                        ),
                    ),
                ),
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
        # (guarded through a list/dict field, as inhabitation requires) is
        # unconditionally finite.
        table = TypeTable()
        table.register(
            TypeDef(
                kind="exception",
                name="Chain",
                module_id=ENTRY_ID,
                fields=(
                    ("code", IntType()),
                    ("causes", ListType(ExceptionType("Chain", module_id=ENTRY_ID))),
                ),
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
                                ),
                            ),
                            module_id=ENTRY_ID,
                        ),
                    ),
                ),
            )
        )
        table.register(
            TypeDef(
                kind="record",
                name="Holder",
                module_id=ENTRY_ID,
                fields=(("p", RecordType("Perfect", type_args=(IntType(),), module_id=ENTRY_ID)),),
            )
        )
        holder = RecordType("Holder", module_id=ENTRY_ID)
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
                                ),
                            ),
                            module_id=ENTRY_ID,
                        ),
                    ),
                ),
            )
        )
        table.register(
            TypeDef(
                kind="record",
                name="Phantom",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(),
            )
        )
        phantom = RecordType(
            "Phantom",
            type_args=(RecordType("Perfect", type_args=(IntType(),), module_id=ENTRY_ID),),
            module_id=ENTRY_ID,
        )
        assert table.has_finite_schema(phantom) is True

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
                        ListType(
                            RecordType("Tree", type_args=(TypeVarType("T"),), module_id=ENTRY_ID)
                        ),
                    ),
                ),
            )
        )
        table.register(
            TypeDef(
                kind="record",
                name="Holder",
                module_id=ENTRY_ID,
                fields=(("t", RecordType("Tree", type_args=(IntType(),), module_id=ENTRY_ID)),),
            )
        )
        holder = RecordType("Holder", module_id=ENTRY_ID)
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
                                ),
                            ),
                            module_id=ENTRY_ID,
                        ),
                    ),
                ),
            )
        )
        result = compute_finite_closure(table)
        perfect_key = (ENTRY_ID, "Perfect")
        assert perfect_key in result.infinite
        assert result.successors[perfect_key] == frozenset(
            {(ENTRY_ID, "Perfect"), (ENTRY_ID, "Pair")}
        )

    def test_caches_result_and_invalidates_on_register(self) -> None:
        table = TypeTable()
        table.register(_pair_def())
        assert table.has_finite_closure(ENTRY_ID, "Pair") is True
        table.unregister(ENTRY_ID, "Pair")
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
                            "Pair", type_args=(ListType(TypeVarType("T")),), module_id=ENTRY_ID
                        ),
                    ),
                ),
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
                            "Q", type_args=(DictType(TypeVarType("T")),), module_id=ENTRY_ID
                        ),
                    ),
                ),
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
                        ),
                    ),
                ),
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "F") is False

    def test_exception_with_base_contributes_no_parameter_edges(self) -> None:
        # The `extends` base is a reference edge like any other, but
        # exceptions are never generic, so it never contributes a
        # parameter-dependency node — the base chain stays finite.
        table = TypeTable()
        table.register(TypeDef(kind="exception", name="Root", module_id=ENTRY_ID))
        table.register(
            TypeDef(
                kind="exception",
                name="Derived",
                module_id=ENTRY_ID,
                fields=(("code", IntType()),),
                base=(ENTRY_ID, "Root"),
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "Root") is True
        assert table.has_finite_closure(ENTRY_ID, "Derived") is True

    def test_dangling_reference_defaults_to_finite(self) -> None:
        # A field referencing a declaration that was never registered (same
        # defensive scenario as the equality-capability fixpoint): the
        # dangling reference must not crash finiteness analysis, and
        # defaults permissively to finite.
        table = TypeTable()
        table.register(
            TypeDef(
                kind="record",
                name="Y",
                module_id=ENTRY_ID,
                fields=(("ghost", RecordType("Ghost", module_id=ENTRY_ID)),),
            )
        )
        assert table.has_finite_closure(ENTRY_ID, "Y") is True

    def test_argument_template_type_var_foreign_to_source_is_ignored(self) -> None:
        # Defensive: an argument template's type variable that is not among
        # the REFERENCING declaration's own type parameters (a malformed
        # template — never produced by the real type builder) is simply
        # ignored rather than crashing, mirroring the equality-capability
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
                            "B",
                            type_args=(TypeVarType("Q"),),
                            module_id=ENTRY_ID,
                        ),
                    ),
                ),
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
                    ("a", RecordType("A", type_args=(TypeVarType("Q"),), module_id=ENTRY_ID)),
                ),
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
                                ),
                            ),
                            module_id=ENTRY_ID,
                        ),
                    ),
                ),
            )
        )
        table.register(
            TypeDef(
                kind="record",
                name="Box",
                module_id=ENTRY_ID,
                type_params=("T",),
                fields=(("value", TypeVarType("T")),),
            )
        )
        root = RecordType(
            "Box",
            type_args=(RecordType("Perfect", type_args=(IntType(),), module_id=ENTRY_ID),),
            module_id=ENTRY_ID,
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
                                ),
                            ),
                            module_id=ENTRY_ID,
                        ),
                    ),
                ),
            )
        )
        root = FunctionType(
            params=(IntType(),),
            result=RecordType("Perfect", type_args=(IntType(),), module_id=ENTRY_ID),
        )
        assert table.has_finite_schema(root) is False

    def test_has_finite_schema_dedupes_repeated_declaration_in_root(self) -> None:
        # X appears TWICE as a sibling type argument of the (finite) root —
        # the reachability walk must not re-process an already-seen
        # declaration key a second time.
        table = TypeTable()
        table.register(_pair_def())
        table.register(
            TypeDef(kind="record", name="X", module_id=ENTRY_ID, fields=(("v", IntType()),))
        )
        x_handle = RecordType("X", module_id=ENTRY_ID)
        root = RecordType("Pair", type_args=(x_handle, x_handle), module_id=ENTRY_ID)
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
                                ),
                            ),
                            module_id=ENTRY_ID,
                        ),
                    ),
                ),
            )
        )
        return table

    def test_first_infinite_declaration_is_none_for_finite_root(self) -> None:
        table = self._perfect_table()
        pair_int = RecordType("Pair", type_args=(IntType(), IntType()), module_id=ENTRY_ID)
        assert table.first_infinite_declaration(IntType()) is None
        assert table.first_infinite_declaration(pair_int) is None

    def test_first_infinite_declaration_names_root_itself(self) -> None:
        table = self._perfect_table()
        perfect_int = RecordType("Perfect", type_args=(IntType(),), module_id=ENTRY_ID)
        assert table.first_infinite_declaration(perfect_int) == (ENTRY_ID, "Perfect")

    def test_first_infinite_declaration_names_culprit_reached_through_field(self) -> None:
        table = self._perfect_table()
        table.register(
            TypeDef(
                kind="record",
                name="Holder",
                module_id=ENTRY_ID,
                fields=(("p", RecordType("Perfect", type_args=(IntType(),), module_id=ENTRY_ID)),),
            )
        )
        holder = RecordType("Holder", module_id=ENTRY_ID)
        assert table.first_infinite_declaration(holder) == (ENTRY_ID, "Perfect")

    def test_no_finite_schema_message_is_none_for_finite_type(self) -> None:
        table = self._perfect_table()
        pair_int = RecordType("Pair", type_args=(IntType(), IntType()), module_id=ENTRY_ID)
        assert table.no_finite_schema_message(pair_int, use="a cast target") is None

    def test_no_finite_schema_message_names_root_when_root_is_the_culprit(self) -> None:
        table = self._perfect_table()
        perfect_int = RecordType("Perfect", type_args=(IntType(),), module_id=ENTRY_ID)
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
                fields=(("p", RecordType("Perfect", type_args=(IntType(),), module_id=ENTRY_ID)),),
            )
        )
        holder = RecordType("Holder", module_id=ENTRY_ID)
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
        mod = ModuleId.from_dotted("mod_a")
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
                                ),
                            ),
                            module_id=mod,
                        ),
                    ),
                ),
            )
        )
        table.register(
            TypeDef(
                kind="record",
                name="Holder",
                module_id=ENTRY_ID,
                fields=(("p", RecordType("Perfect", type_args=(IntType(),), module_id=mod)),),
            )
        )
        holder = RecordType("Holder", module_id=ENTRY_ID)
        message = table.no_finite_schema_message(holder, use="a parameter type")
        assert message is not None
        assert "mod_a::Perfect" in message
        assert "a parameter type" in message
