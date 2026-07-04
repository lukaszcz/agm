"""Tests for the shared nominal type-declaration table.

Covers ``TypeDef``/``TypeTable`` in ``agm.agl.semantics.type_table`` and the
dual-write path that populates it alongside the embedded
``RecordType``/``EnumType`` representation: the type builder
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
from agm.agl.semantics.type_table import TypeDef, TypeTable, create_seeded_type_table
from agm.agl.semantics.types import (
    BUILTIN_PRELUDE_TYPES,
    DictType,
    EnumType,
    IntType,
    ListType,
    RecordType,
    TextType,
    TypeVarType,
)
from agm.agl.syntax.nodes import LetDecl, VarDecl
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
        handle = RecordType(name="Point", fields={}, module_id=ENTRY_ID)
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
        handle = EnumType(name="Color", variants={}, module_id=ENTRY_ID)
        result = table.enum_variants(handle)
        assert {v: dict(f) for v, f in result.items()} == {
            "Red": {},
            "Custom": {"hex": TextType()},
        }

    def test_record_fields_missing_def_raises_keyerror(self) -> None:
        table = TypeTable()
        handle = RecordType(name="Ghost", fields={}, module_id=ENTRY_ID)
        with pytest.raises(KeyError):
            table.record_fields(handle)

    def test_enum_variants_missing_def_raises_keyerror(self) -> None:
        table = TypeTable()
        handle = EnumType(name="Ghost", variants={}, module_id=ENTRY_ID)
        with pytest.raises(KeyError):
            table.enum_variants(handle)

    def test_record_fields_raises_when_key_registered_as_enum(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(kind="enum", name="Color", module_id=ENTRY_ID, variants=(("Red", ()),))
        )
        handle = RecordType(name="Color", fields={}, module_id=ENTRY_ID)
        with pytest.raises(AssertionError):
            table.record_fields(handle)

    def test_enum_variants_raises_when_key_registered_as_record(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(kind="record", name="Point", module_id=ENTRY_ID, fields=(("x", IntType()),))
        )
        handle = EnumType(name="Point", variants={}, module_id=ENTRY_ID)
        with pytest.raises(AssertionError):
            table.enum_variants(handle)


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
            name="Pair", fields={}, type_args=(IntType(), TextType()), module_id=ENTRY_ID
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
            name="Maybe", variants={}, type_args=(IntType(),), module_id=ENTRY_ID
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
        handle = RecordType(name="Point", fields={}, module_id=ENTRY_ID)
        first = table.record_fields(handle)
        second = table.record_fields(handle)
        assert first is second

    def test_enum_variants_returns_same_object_for_same_handle(self) -> None:
        table = TypeTable()
        table.register(
            TypeDef(kind="enum", name="Color", module_id=ENTRY_ID, variants=(("Red", ()),))
        )
        handle = EnumType(name="Color", variants={}, module_id=ENTRY_ID)
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
            name="Box", fields={}, type_args=(IntType(),), module_id=ENTRY_ID
        )
        text_handle = RecordType(
            name="Box", fields={}, type_args=(TextType(),), module_id=ENTRY_ID
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
            name="Maybe", variants={}, type_args=(IntType(),), module_id=ENTRY_ID
        )
        text_handle = EnumType(
            name="Maybe", variants={}, type_args=(TextType(),), module_id=ENTRY_ID
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
        handle = RecordType(name="Point", fields={}, module_id=ENTRY_ID)
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
        handle = EnumType(name="Color", variants={}, module_id=ENTRY_ID)
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
        handle = RecordType(name="Shared", fields={}, module_id=ENTRY_ID)
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
        handle = RecordType(name="X", fields={}, module_id=ENTRY_ID)
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
            if isinstance(typ, RecordType):
                handle = RecordType(name=name, fields={}, module_id=PRELUDE_ID)
                assert dict(table.record_fields(handle)) == dict(typ.fields)
            else:
                handle = EnumType(name=name, variants={}, module_id=PRELUDE_ID)
                result = table.enum_variants(handle)
                assert {v: dict(f) for v, f in result.items()} == {
                    v: dict(f) for v, f in typ.variants.items()
                }

    def test_generic_option_seeded_under_std_core(self) -> None:
        table = create_seeded_type_table()
        typedef = table.get(STD_CORE_ID, "Option")
        assert typedef is not None
        assert typedef.type_params == ("T",)
        handle = EnumType(
            name="Option", variants={}, type_args=(TextType(),), module_id=STD_CORE_ID
        )
        result = table.enum_variants(handle)
        assert {v: dict(f) for v, f in result.items()} == {
            "None": {},
            "Some": {"value": TextType()},
        }


# ---------------------------------------------------------------------------
# Dual-write consistency: single-module
# ---------------------------------------------------------------------------


class TestDualWriteSingleModule:
    def test_non_generic_record(self) -> None:
        checked = _check(
            "record Point\n  x: int\n  y: int\nlet p = Point(x = 1, y = 2)\np"
        )
        point = checked.type_env.get_type("Point")
        assert isinstance(point, RecordType)
        table = checked.type_env.type_table
        assert dict(table.record_fields(point)) == dict(point.fields)

    def test_generic_record(self) -> None:
        checked = _check(
            "record Box[T]\n  value: T\nlet b: Box[int] = Box(value = 1)\nb"
        )
        box = _binding_value_type(checked, "b")
        assert isinstance(box, RecordType)
        table = checked.type_env.type_table
        assert dict(table.record_fields(box)) == dict(box.fields)

    def test_non_generic_enum(self) -> None:
        checked = _check(
            "enum Color\n  | Red\n  | Green\n  | Blue\nlet c = Red\nc"
        )
        color = checked.type_env.get_type("Color")
        assert isinstance(color, EnumType)
        table = checked.type_env.type_table
        result = table.enum_variants(color)
        assert {v: dict(f) for v, f in result.items()} == {
            v: dict(f) for v, f in color.variants.items()
        }

    def test_generic_enum(self) -> None:
        checked = _check(
            "enum Maybe[T]\n  | none\n  | just(value: T)\nlet m = just(value = 1)\nm"
        )
        maybe = _binding_value_type(checked, "m")
        assert isinstance(maybe, EnumType)
        table = checked.type_env.type_table
        result = table.enum_variants(maybe)
        assert {v: dict(f) for v, f in result.items()} == {
            v: dict(f) for v, f in maybe.variants.items()
        }


# ---------------------------------------------------------------------------
# Dual-write consistency: graph mode
# ---------------------------------------------------------------------------


class TestDualWriteGraphMode:
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
        assert dict(mylib_table.record_fields(point)) == dict(point.fields)

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
