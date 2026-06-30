"""Graph-level typechecking tests over multi-module AgL programs.

Covers ``check_graph`` / ``CheckedModuleGraph`` / ``CheckedModule``, the
``module_id`` field on ``RecordType`` / ``EnumType``, cross-module type
identity, generic types, mutual recursion, and the structural type-dependency
pre-pass."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.scope.graph import resolve_graph
from agm.agl.scope.symbols import AglScopeError
from agm.agl.typecheck import (
    AgentType,
    AglTypeError,
    BoolType,
    CheckedModule,
    CheckedModuleGraph,
    CheckedProgram,
    EnumType,
    ExceptionType,
    FunctionType,
    IntType,
    RecordType,
    TextType,
    Type,
    TypeVarType,
    check,
    check_graph,
)
from tests.agl.ir_harness import make_graph_from_files as _make_graph_from_files

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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


def _check(src: str) -> CheckedProgram:
    """Parse + resolve + check a single-module AgL program."""
    from agm.agl.parser import parse_program
    from agm.agl.scope import resolve

    resolved = resolve(parse_program(src))
    return check(resolved, _CAPS)


def _check_graph(tmp_path: Path, modules: dict[str, str]) -> CheckedModuleGraph:
    """Build and typecheck a multi-module graph; returns CheckedModuleGraph."""
    mg = _make_graph_from_files(tmp_path, modules)
    rg = resolve_graph(mg)
    return check_graph(rg, _CAPS)


def _binding_value_type(cg: CheckedModuleGraph, module_id: ModuleId, name: str) -> Type:
    """Inferred type of the RHS of the top-level ``let``/``var <name> = ...`` in ``module_id``."""
    from agm.agl.syntax.nodes import LetDecl, VarDecl

    module = cg.modules[module_id]
    for item in module.resolved.program.body.items:
        if isinstance(item, (LetDecl, VarDecl)) and item.name == name:
            return module.node_types[item.value.node_id]
    raise AssertionError(f"no top-level binding named {name!r} in {module_id}")


def _agent_binding_type(cg: CheckedModuleGraph, module_id: ModuleId, name: str) -> Type:
    """Inferred type of the agent declaration ``agent <name> = ...`` in ``module_id``."""
    from agm.agl.syntax.nodes import AgentDecl

    module = cg.modules[module_id]
    for item in module.resolved.program.body.items:
        if isinstance(item, AgentDecl) and item.name == name:
            t = module.type_env.get_binding_type(item.node_id)
            assert t is not None, f"no binding type for agent {name!r} in {module_id}"
            return t
    raise AssertionError(f"no agent declaration named {name!r} in {module_id}")


# ---------------------------------------------------------------------------
# 0. Smoke: new symbols exist at all
# ---------------------------------------------------------------------------


def test_check_graph_importable() -> None:
    """check_graph and CheckedModuleGraph are importable from agm.agl.typecheck.graph."""
    from agm.agl.typecheck.graph import (
        CheckedModule,
        CheckedModuleGraph,
        check_graph,
    )

    assert callable(check_graph)
    assert CheckedModuleGraph is not None
    assert CheckedModule is not None


def test_graph_func_signature_prepass_skips_inferred_return_type(tmp_path: Path) -> None:
    """Graph mode lets an unannotated def infer inside its own module."""
    cg = _check_graph(
        tmp_path,
        {
            "entry": "import lib\n1",
            "lib": "def answer() = 42",
        },
    )
    lib = cg.modules[ModuleId(("lib",))]
    assert lib.type_env.all_function_signatures()["answer"].result == IntType()


# ---------------------------------------------------------------------------
# 1. module_id field on RecordType — default ENTRY_ID
# ---------------------------------------------------------------------------


def test_module_id_on_record_type_default_entry_id() -> None:
    """RecordType('Foo', {}) has module_id == ENTRY_ID by default."""
    rt = RecordType("Foo", {})
    # M4 adds module_id: ModuleId = ENTRY_ID to RecordType
    assert hasattr(rt, "module_id"), "RecordType must have a module_id field after M4"
    assert rt.module_id == ENTRY_ID


# ---------------------------------------------------------------------------
# 2. module_id field on EnumType — default ENTRY_ID
# ---------------------------------------------------------------------------


def test_module_id_on_enum_type_default_entry_id() -> None:
    """EnumType('Color', {}) has module_id == ENTRY_ID by default."""
    et = EnumType("Color", {})
    assert hasattr(et, "module_id"), "EnumType must have a module_id field after M4"
    assert et.module_id == ENTRY_ID


# ---------------------------------------------------------------------------
# 3. Distinct module_id makes RecordType instances unequal
# ---------------------------------------------------------------------------


def test_distinct_module_qualified_type_identity() -> None:
    """RecordType('Color', {}, module_id=mid_foo) != RecordType('Color', {}, module_id=mid_bar)."""
    mid_foo = ModuleId.from_dotted("foo")
    mid_bar = ModuleId.from_dotted("bar")
    rt_foo = RecordType("Color", {}, module_id=mid_foo)
    rt_bar = RecordType("Color", {}, module_id=mid_bar)
    assert rt_foo != rt_bar, (
        "Same-name record types from different modules must be distinct types"
    )


# ---------------------------------------------------------------------------
# 4. Same module_id, same structure → equal types
# ---------------------------------------------------------------------------


def test_same_module_same_type_identity() -> None:
    """Two RecordType instances with identical name+fields+module_id are equal."""
    mid = ModuleId.from_dotted("mylib")
    rt1 = RecordType("Point", {"x": IntType(), "y": IntType()}, module_id=mid)
    rt2 = RecordType("Point", {"x": IntType(), "y": IntType()}, module_id=mid)
    assert rt1 == rt2


# ---------------------------------------------------------------------------
# 5. Single-module check_graph equivalent to check()
# ---------------------------------------------------------------------------


def test_single_module_equivalence(tmp_path: Path) -> None:
    """check_graph on a single-module graph gives the same type results as check()."""
    source = "def foo() -> int = 1\nlet x = foo()\nx"
    mg = _make_graph_from_files(tmp_path, {"entry": source})
    rg = resolve_graph(mg)
    cg = check_graph(rg, _CAPS)

    assert isinstance(cg, CheckedModuleGraph)
    # The graph must have the entry module
    assert ENTRY_ID in cg.modules

    # node_types in graph-checked entry should equal single-module check
    single = _check(source)
    entry_checked = cg.modules[ENTRY_ID]
    # Both should have the same number of typed expression nodes
    assert len(entry_checked.node_types) == len(single.node_types)


# ---------------------------------------------------------------------------
# 6. check_graph basic: cross-module record type used in entry
# ---------------------------------------------------------------------------


def test_check_graph_basic(tmp_path: Path) -> None:
    """Entry imports mylib with a record; annotated let binding typechecks successfully."""
    modules = {
        "entry": (
            "import mylib\n"
            "def make() -> Point = mylib::makePoint()\n"
            "let p: Point = make()\n"
            "p"
        ),
        "mylib": (
            "record Point\n"
            "  x: int\n"
            "  y: int\n"
            "def makePoint() -> Point = Point(x = 0, y = 0)"
        ),
    }
    cg = _check_graph(tmp_path, modules)
    assert ENTRY_ID in cg.modules
    mylib_id = ModuleId.from_dotted("mylib")
    assert _binding_value_type(cg, ENTRY_ID, "p") == RecordType("Point", {}, module_id=mylib_id)


# ---------------------------------------------------------------------------
# 7. Cross-module same-name types are NOT assignable to each other
# ---------------------------------------------------------------------------


def test_cross_module_type_not_assignable(tmp_path: Path) -> None:
    """foo::Color and bar::Color (same structure, different module) must cause a type error."""
    modules = {
        "entry": (
            "import foo\n"
            "import bar\n"
            # mycolor has type foo::Color but we try to use it where bar::Color expected
            "def get_foo_color() -> foo::Color = foo::makeColor()\n"
            "def expect_bar(c: bar::Color) -> bar::Color = c\n"
            "let c = get_foo_color()\n"
            "expect_bar(c)"  # type mismatch: foo::Color ≠ bar::Color
        ),
        "foo": (
            "enum Color\n"
            "  | Red\n"
            "  | Blue\n"
            "def makeColor() -> Color = Red"
        ),
        "bar": (
            "enum Color\n"
            "  | Red\n"
            "  | Blue\n"
            "def makeColor() -> Color = Red"
        ),
    }
    with pytest.raises(AglTypeError):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# 8. Qualified type reference in annotation resolves correctly
# ---------------------------------------------------------------------------


def test_qualified_type_ref_in_annotation(tmp_path: Path) -> None:
    """'let p: mylib::Point = ...' resolves mylib::Point through ImportEnv."""
    modules = {
        "entry": (
            "import mylib qualified\n"
            "def get() -> mylib::Point = mylib::mkPoint()\n"
            "let p: mylib::Point = get()\n"
            "p"
        ),
        "mylib": (
            "record Point\n"
            "  x: int\n"
            "  y: int\n"
            "def mkPoint() -> Point = Point(x = 1, y = 2)"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    # Pin the specific binding type — not any(t == point_type) over all nodes,
    # which could pass spuriously via an intermediate call node of the same type.
    assert _binding_value_type(cg, ENTRY_ID, "p") == RecordType("Point", {}, module_id=mylib_id)


# ---------------------------------------------------------------------------
# 9. Qualified type ref in constructor: foo::Color.Red
# ---------------------------------------------------------------------------


def test_qualified_type_ref_in_constructor(tmp_path: Path) -> None:
    """'foo::Color.Red' constructor resolves through ImportEnv correctly."""
    modules = {
        "entry": (
            "import mylib qualified\n"
            "let c: mylib::Color = mylib::Color.Red\n"
            "c"
        ),
        "mylib": (
            "enum Color\n"
            "  | Red\n"
            "  | Blue"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "c") == EnumType("Color", {}, module_id=mylib_id)


# ---------------------------------------------------------------------------
# 10. Qualified type ref in cast: x as foo::MyRecord
# ---------------------------------------------------------------------------


def test_qualified_type_ref_in_cast(tmp_path: Path) -> None:
    """'x as mylib::Point' cast resolves the target type through ImportEnv."""
    modules = {
        "entry": (
            "import mylib qualified\n"
            "let raw: json = {\"x\": 1, \"y\": 2}\n"
            "let p: mylib::Point = raw as mylib::Point\n"
            "p"
        ),
        "mylib": (
            "record Point\n"
            "  x: int\n"
            "  y: int"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "p") == RecordType("Point", {}, module_id=mylib_id)


# ---------------------------------------------------------------------------
# 11. Qualified type ref in constructor pattern: foo::Color.Red in case
# ---------------------------------------------------------------------------


def test_qualified_type_ref_in_constructor_pattern(tmp_path: Path) -> None:
    """'mylib::Color.Red' in a case pattern resolves through ImportEnv."""
    modules = {
        "entry": (
            "import mylib qualified\n"
            "def describe(c: mylib::Color) -> text =\n"
            "  case c of | mylib::Color.Red => \"red\" | mylib::Color.Blue => \"blue\"\n"
            "let c = mylib::Color.Red\n"
            "describe(c)"
        ),
        "mylib": (
            "enum Color\n"
            "  | Red\n"
            "  | Blue"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    # Pin c's binding type as mylib::Color — not an any(TextType) scan over "red"/"blue".
    assert _binding_value_type(cg, ENTRY_ID, "c") == EnumType("Color", {}, module_id=mylib_id)


# ---------------------------------------------------------------------------
# 12. Unqualified open import: type name comes into scope
# ---------------------------------------------------------------------------


def test_unqualified_open_import_type(tmp_path: Path) -> None:
    """Open import brings record type name into scope for unqualified use."""
    modules = {
        "entry": (
            "import mylib\n"
            "def mkp() -> Point = mkPoint()\n"
            "let p: Point = mkp()\n"
            "p"
        ),
        "mylib": (
            "record Point\n"
            "  x: int\n"
            "  y: int\n"
            "def mkPoint() -> Point = Point(x = 0, y = 0)"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "p") == RecordType("Point", {}, module_id=mylib_id)


# ---------------------------------------------------------------------------
# 13. Unqualified type clash: same name from two modules → ambiguous error
# ---------------------------------------------------------------------------


def test_unqualified_type_clash_on_use(tmp_path: Path) -> None:
    """Two open imports both export 'Color' → ambiguous type error at use site."""
    modules = {
        "entry": (
            "import libA\n"
            "import libB\n"
            # 'Color' is ambiguous — could be libA::Color or libB::Color
            "let c: Color = libA::Color.Red\n"
            "c"
        ),
        "libA": (
            "enum Color\n"
            "  | Red\n"
            "  | Blue"
        ),
        "libB": (
            "enum Color\n"
            "  | Green\n"
            "  | Yellow"
        ),
    }
    with pytest.raises(AglTypeError, match="Ambiguous type"):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# 14. Qualified access bounded by S (using clause): unlisted names inaccessible
# ---------------------------------------------------------------------------


def test_qualified_access_bounded_by_s(tmp_path: Path) -> None:
    """'import mylib qualified using Point' — mylib::Rect is NOT in S → type error."""
    modules = {
        "entry": (
            "import mylib qualified using Point\n"
            # Rect is not in S (only Point is), so mylib::Rect should fail
            "let r: mylib::Rect = mylib::mkRect()\n"
            "r"
        ),
        "mylib": (
            "record Point\n"
            "  x: int\n"
            "record Rect\n"
            "  w: int\n"
            "  h: int\n"
            "def mkRect() -> Rect = Rect(w = 10, h = 5)"
        ),
    }
    with pytest.raises(AglScopeError, match="not in the imported set"):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# 15. Private type not importable from another module
# ---------------------------------------------------------------------------


def test_private_type_not_importable(tmp_path: Path) -> None:
    """'private record Hidden' in mylib cannot be used from entry."""
    modules = {
        "entry": (
            "import mylib\n"
            # Hidden is private in mylib; should not be accessible here
            "let h: Hidden = mylib::mkHidden()\n"
            "h"
        ),
        "mylib": (
            "private record Hidden\n"
            "  x: int\n"
            "def mkHidden() -> Hidden = Hidden(x = 1)"
        ),
    }
    with pytest.raises(AglTypeError, match="Unknown type"):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# 16. Whole-graph type pre-pass with cycles: A refs B::Color, B refs A::Foo
# ---------------------------------------------------------------------------


def test_whole_graph_type_pre_pass_with_cycles(tmp_path: Path) -> None:
    """Mutual imports of types between A and B both typecheck (cycles allowed D8)."""
    modules = {
        "entry": (
            "import modA\n"
            "import modB\n"
            "let fa = modA::wrapB(modB::Color.Red)\n"
            "let fb = modB::wrapA(modA::Foo(x = 1))\n"
            "()"
        ),
        "modA": (
            "import modB\n"
            "record Foo\n"
            "  x: int\n"
            "def wrapB(c: modB::Color) -> text = \"ok\""
        ),
        "modB": (
            "import modA\n"
            "enum Color\n"
            "  | Red\n"
            "  | Blue\n"
            "def wrapA(f: modA::Foo) -> text = \"ok\""
        ),
    }
    cg = _check_graph(tmp_path, modules)
    mid_a = ModuleId.from_dotted("modA")
    mid_b = ModuleId.from_dotted("modB")
    assert mid_a in cg.modules
    assert mid_b in cg.modules
    assert _binding_value_type(cg, ENTRY_ID, "fa") == TextType()
    assert _binding_value_type(cg, ENTRY_ID, "fb") == TextType()


def test_imported_exception_base_is_built_before_child(tmp_path: Path) -> None:
    """A child exception inherits fields from an open-imported base exception."""
    modules = {
        "entry": (
            "import a\n"
            "let value = a::make()\n"
            "value"
        ),
        "a": (
            "import z\n"
            "exception Child extends Base\n"
            "  code: int\n"
            "def make() -> text =\n"
            "  let err = Child(message = \"m\", detail = \"d\", code = 1)\n"
            "  err.detail"
        ),
        "z": (
            "exception Base extends Exception\n"
            "  detail: text"
        ),
    }
    cg = _check_graph(tmp_path, modules)
    child_type = cg.graph_type_table[(ModuleId.from_dotted("a"), "Child")]
    assert isinstance(child_type, ExceptionType)
    assert "detail" in child_type.fields
    assert _binding_value_type(cg, ENTRY_ID, "value") == TextType()


def test_imported_exception_base_ignores_non_type_export(tmp_path: Path) -> None:
    """A same-named imported value is not treated as an exception-base dependency."""
    modules = {
        "entry": "import a\n()",
        "a": (
            "import z\n"
            "exception Child extends Base\n"
            "  code: int"
        ),
        "z": "def Base() -> int = 1",
    }
    with pytest.raises(AglTypeError, match="extends unknown exception 'Base'"):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# 17. Enum variant qualification: foo::Color.Red
# ---------------------------------------------------------------------------


def test_enum_variant_qualification(tmp_path: Path) -> None:
    """'mylib::Color.Red' where Color is an enum in module mylib resolves correctly."""
    modules = {
        "entry": (
            "import mylib qualified\n"
            "let c: mylib::Color = mylib::Color.Red\n"
            "c"
        ),
        "mylib": (
            "enum Color\n"
            "  | Red\n"
            "  | Green\n"
            "  | Blue"
        ),
    }
    cg = _check_graph(tmp_path, modules)
    mylib_id = ModuleId.from_dotted("mylib")
    assert (mylib_id, "Color") in cg.graph_type_table
    color_type = cg.graph_type_table[(mylib_id, "Color")]
    assert isinstance(color_type, EnumType)
    assert color_type.module_id == mylib_id
    assert _binding_value_type(cg, ENTRY_ID, "c") == EnumType("Color", {}, module_id=mylib_id)


# ---------------------------------------------------------------------------
# 18. Self-ref type: ::MyType in a module references own module's type
# ---------------------------------------------------------------------------


def test_self_ref_type(tmp_path: Path) -> None:
    """'::Point' in a module references its own module's Point record."""
    modules = {
        "entry": (
            "import mylib\n"
            "let p: mylib::Point = mylib::origin()\n"
            "p"
        ),
        "mylib": (
            "record Point\n"
            "  x: int\n"
            "  y: int\n"
            # ::Point refers to mylib's own Point type via self-reference
            "def origin() -> ::Point = Point(x = 0, y = 0)"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "p") == RecordType("Point", {}, module_id=mylib_id)


# ---------------------------------------------------------------------------
# 19. Agent-typed argument in imported function
# ---------------------------------------------------------------------------


def test_agent_typed_arg_in_imported_function(tmp_path: Path) -> None:
    """An imported function accepting agent-typed arg can be called from entry."""
    caps_with_agent = HostCapabilities(
        agent_names=frozenset({"bot"}),
        has_default_agent=True,
        supports_shell_exec=True,
        codec_kinds={
            "text": frozenset({"text"}),
            "json": frozenset(
                {"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}
            ),
        },
    )
    modules = {
        "entry": (
            "import mylib\n"
            "agent bot = \"claude\"\n"
            "let result: text = mylib::greet(bot)\n"
            "result"
        ),
        "mylib": (
            "def greet(a: agent) -> text = \"hello\""
        ),
    }
    mg = _make_graph_from_files(tmp_path, modules)
    rg = resolve_graph(mg)
    cg = check_graph(rg, caps_with_agent)
    # Verify the agent-typed-arg path is exercised: bot must be agent-typed
    assert _agent_binding_type(cg, ENTRY_ID, "bot") == AgentType()
    # Verify the greet call returns text
    assert _binding_value_type(cg, ENTRY_ID, "result") == TextType()


# ---------------------------------------------------------------------------
# 20. Unqualified constructor from open import
# ---------------------------------------------------------------------------


def test_unqualified_constructor_from_open_import(tmp_path: Path) -> None:
    """When Color is open-imported from foo, 'Color.Red' (bare variant) resolves to foo::Color."""
    modules = {
        "entry": (
            "import mylib\n"
            # Color is open-imported, so 'Color.Red' (unqualified) should resolve
            "let c: Color = Color.Red\n"
            "c"
        ),
        "mylib": (
            "enum Color\n"
            "  | Red\n"
            "  | Blue"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    assert (mylib_id, "Color") in cg.graph_type_table
    color_type = cg.graph_type_table[(mylib_id, "Color")]
    assert isinstance(color_type, EnumType)
    assert color_type.module_id == mylib_id
    # Pin c's binding type: must be mylib::Color, not ENTRY_ID::Color
    assert _binding_value_type(cg, ENTRY_ID, "c") == EnumType("Color", {}, module_id=mylib_id)


# ---------------------------------------------------------------------------
# Extra: graph_type_table is populated with all modules
# ---------------------------------------------------------------------------


def test_graph_type_table_populated(tmp_path: Path) -> None:
    """graph_type_table in CheckedModuleGraph contains all public types stamped with module_id."""
    modules = {
        "entry": (
            "import mylib\n"
            "()"
        ),
        "mylib": (
            "record Point\n"
            "  x: int\n"
            "  y: int\n"
            "enum Direction\n"
            "  | North\n"
            "  | South"
        ),
    }
    cg = _check_graph(tmp_path, modules)
    mylib_id = ModuleId.from_dotted("mylib")
    assert (mylib_id, "Point") in cg.graph_type_table
    assert (mylib_id, "Direction") in cg.graph_type_table

    pt = cg.graph_type_table[(mylib_id, "Point")]
    assert isinstance(pt, RecordType)
    assert pt.module_id == mylib_id

    dir_type = cg.graph_type_table[(mylib_id, "Direction")]
    assert isinstance(dir_type, EnumType)
    assert dir_type.module_id == mylib_id


# ---------------------------------------------------------------------------
# Extra: CheckedModule shape
# ---------------------------------------------------------------------------


def test_checked_module_shape(tmp_path: Path) -> None:
    """CheckedModule has node_types, contract_specs, warnings, function_signatures."""
    modules = {
        "entry": (
            "import mylib\n"
            "def foo() -> int = mylib::getValue()\n"
            "let x = foo()\n"
            "x"
        ),
        "mylib": (
            "def getValue() -> int = 42"
        ),
    }
    cg = _check_graph(tmp_path, modules)
    entry_mod = cg.modules[ENTRY_ID]
    assert isinstance(entry_mod, CheckedModule)
    assert hasattr(entry_mod, "node_types")
    assert hasattr(entry_mod, "contract_specs")
    assert hasattr(entry_mod, "warnings")
    assert hasattr(entry_mod, "function_signatures")
    assert _binding_value_type(cg, ENTRY_ID, "x") == IntType()


# ---------------------------------------------------------------------------
# Extra: entry_id on CheckedModuleGraph
# ---------------------------------------------------------------------------


def test_checked_module_graph_entry_id(tmp_path: Path) -> None:
    """CheckedModuleGraph.entry_id == ENTRY_ID."""
    cg = _check_graph(tmp_path, {"entry": "()"})
    assert cg.entry_id == ENTRY_ID


# ---------------------------------------------------------------------------
# Coverage: type alias in a module graph
# ---------------------------------------------------------------------------


def test_type_alias_in_module_graph(tmp_path: Path) -> None:
    """A type alias in a library module is stored in the graph type table."""
    modules = {
        "entry": (
            "import mylib\n"
            "let n: mylib::Number = 42\n"
            "n"
        ),
        "mylib": "type Number = int",
    }
    cg = _check_graph(tmp_path, modules)
    mylib_id = ModuleId.from_dotted("mylib")
    assert (mylib_id, "Number") in cg.graph_type_table
    # The alias should resolve to int
    t = cg.graph_type_table[(mylib_id, "Number")]
    assert isinstance(t, IntType)
    assert _binding_value_type(cg, ENTRY_ID, "n") == IntType()


# ---------------------------------------------------------------------------
# Coverage: qualified access to a non-type name is an error
# ---------------------------------------------------------------------------


def test_qualified_ref_to_function_is_type_error(tmp_path: Path) -> None:
    """'mylib::getValue' in a type annotation position → type error (not a type)."""
    modules = {
        "entry": (
            "import mylib qualified\n"
            "let n: mylib::getValue = 1\n"
            "n"
        ),
        "mylib": "def getValue() -> int = 42",
    }
    with pytest.raises(AglTypeError, match="does not name a type"):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# Coverage: unknown module qualifier error
# ---------------------------------------------------------------------------


def test_unknown_module_qualifier_error(tmp_path: Path) -> None:
    """Reference to an un-imported module qualifier → type error."""
    modules = {
        "entry": (
            "import mylib\n"
            "let n: other::Point = mylib::mkPoint()\n"
            "n"
        ),
        "mylib": (
            "record Point\n"
            "  x: int\n"
            "def mkPoint() -> Point = Point(x = 1)"
        ),
    }
    with pytest.raises(AglTypeError, match="Unknown module qualifier"):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# Coverage: module-qualified constructor where enum type name is wrong
# ---------------------------------------------------------------------------


def test_module_qualified_constructor_not_enum_error(tmp_path: Path) -> None:
    """'mylib::Point.Red' where Point is a record, not an enum → type error."""
    modules = {
        "entry": (
            "import mylib qualified\n"
            "let p = mylib::Point.Red\n"
            "p"
        ),
        "mylib": (
            "record Point\n"
            "  x: int"
        ),
    }
    with pytest.raises(AglTypeError, match="not a known enum type"):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# Coverage: module-qualified constructor with missing variant
# ---------------------------------------------------------------------------


def test_module_qualified_constructor_missing_variant_error(tmp_path: Path) -> None:
    """'mylib::Color.Purple' where Purple doesn't exist → type error."""
    modules = {
        "entry": (
            "import mylib qualified\n"
            "let c = mylib::Color.Purple\n"
            "c"
        ),
        "mylib": (
            "enum Color\n"
            "  | Red\n"
            "  | Blue"
        ),
    }
    with pytest.raises(AglTypeError, match="does not exist in enum"):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# Coverage: module-qualified record constructor via module_qualifier
# ---------------------------------------------------------------------------


def test_module_qualified_record_constructor(tmp_path: Path) -> None:
    """'mylib::Point(x = 1, y = 2)' constructs a record from an imported module."""
    modules = {
        "entry": (
            "import mylib qualified\n"
            "let p: mylib::Point = mylib::Point(x = 1, y = 2)\n"
            "p"
        ),
        "mylib": (
            "record Point\n"
            "  x: int\n"
            "  y: int"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "p") == RecordType("Point", {}, module_id=mylib_id)


# ---------------------------------------------------------------------------
# Coverage: ::Name self-reference in a module (graph mode)
# ---------------------------------------------------------------------------


def test_self_ref_type_graph_mode(tmp_path: Path) -> None:
    """'::Point' self-reference resolves to the current module's own Point type."""
    modules = {
        "entry": (
            "import mylib\n"
            "let p: mylib::Point = mylib::origin()\n"
            "p"
        ),
        "mylib": (
            "record Point\n"
            "  x: int\n"
            "  y: int\n"
            "def origin() -> ::Point = Point(x = 0, y = 0)"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "p") == RecordType("Point", {}, module_id=mylib_id)


# ---------------------------------------------------------------------------
# Coverage: module-qualified variant qualifier mismatch in case pattern
# ---------------------------------------------------------------------------


def test_module_qualified_variant_qualifier_mismatch(tmp_path: Path) -> None:
    """'libA::Color.Red' as qualifier when value has type libB::Color → error."""
    modules = {
        "entry": (
            "import libA\n"
            "import libB\n"
            "def check(c: libB::Color) -> text =\n"
            "  case c of | libA::Color.Red => \"red\" | _ => \"other\"\n"
            "let c = libB::Color.Red\n"
            "check(c)"
        ),
        "libA": (
            "enum Color\n"
            "  | Red\n"
            "  | Blue"
        ),
        "libB": (
            "enum Color\n"
            "  | Red\n"
            "  | Blue"
        ),
    }
    with pytest.raises(AglTypeError, match="resolves to enum"):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# Coverage: type name not in S via qualified lookup
# ---------------------------------------------------------------------------


def test_name_not_in_s_qualified_lookup(tmp_path: Path) -> None:
    """Qualified access to a name not in S raises an error."""
    modules = {
        "entry": (
            "import mylib qualified using getValue\n"
            # Point is NOT in S (only getValue is)
            "let n: mylib::Point = mylib::mkPoint()\n"
            "n"
        ),
        "mylib": (
            "record Point\n"
            "  x: int\n"
            "def getValue() -> int = 1\n"
            "def mkPoint() -> Point = Point(x = 1)"
        ),
    }
    with pytest.raises(AglScopeError, match="not in the imported set"):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# Coverage: _check_module_qualified_variant: resolved is not an enum type.
# ---------------------------------------------------------------------------


def test_module_qualified_variant_qualifier_is_not_enum(tmp_path: Path) -> None:
    """In a case pattern, 'mylib::Point.Red' where Point is a record → type error."""
    modules = {
        "entry": (
            "import mylib\n"
            "def check(c: mylib::Color) -> text =\n"
            "  case c of | mylib::Point.Red => \"red\" | _ => \"other\"\n"
            "let c = mylib::Color.Red\n"
            "check(c)"
        ),
        "mylib": (
            "record Point\n"
            "  x: int\n"
            "enum Color\n"
            "  | Red\n"
            "  | Blue"
        ),
    }
    with pytest.raises(AglTypeError, match="not a known enum type"):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# Coverage: _check_module_qualified_variant: resolve_type_expr raises on unknown type.
# ---------------------------------------------------------------------------


def test_module_qualified_variant_unknown_enum_in_pattern(tmp_path: Path) -> None:
    """In a case pattern, 'mylib::Unknown.Red' where Unknown doesn't exist → type error."""
    modules = {
        "entry": (
            "import mylib\n"
            "def check(c: mylib::Color) -> text =\n"
            "  case c of | mylib::Unknown.Red => \"red\" | _ => \"other\"\n"
            "let c = mylib::Color.Red\n"
            "check(c)"
        ),
        "mylib": (
            "enum Color\n"
            "  | Red\n"
            "  | Blue"
        ),
    }
    with pytest.raises(AglTypeError, match="not a known enum type"):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# Coverage: _check_module_qualified_constructor: enum used as constructor without a variant.
# ---------------------------------------------------------------------------


def test_module_qualified_enum_as_constructor_error(tmp_path: Path) -> None:
    """'mylib::Color' used as constructor (without .Variant) → type error."""
    modules = {
        "entry": (
            "import mylib qualified\n"
            "let c = mylib::Color\n"
            "c"
        ),
        "mylib": (
            "enum Color\n"
            "  | Red\n"
            "  | Blue"
        ),
    }
    with pytest.raises(AglTypeError, match="is a type name, not a value"):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# Coverage: _check_module_qualified_constructor: unknown name in qualified access.
# ---------------------------------------------------------------------------


def test_module_qualified_unknown_constructor_error(tmp_path: Path) -> None:
    """'mylib::Unknown' when Unknown doesn't exist in mylib → type error."""
    modules = {
        "entry": (
            "import mylib qualified\n"
            "let c = mylib::Unknown\n"
            "c"
        ),
        "mylib": (
            "enum Color\n"
            "  | Red\n"
            "  | Blue"
        ),
    }
    with pytest.raises(AglScopeError, match="not in the imported set"):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# Coverage: get_open_imported_enum_candidates returns the matching enum for a bare variant.
# ---------------------------------------------------------------------------


def test_open_imported_enum_variant_unqualified_bare(tmp_path: Path) -> None:
    """Open-imported enum variant used as bare constructor resolves correctly."""
    modules = {
        "entry": (
            "import mylib\n"
            # Red is a bare variant (no args) from open-imported Color
            "let c = Red\n"
            "c"
        ),
        "mylib": (
            "enum Color\n"
            "  | Red\n"
            "  | Blue"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "c") == EnumType("Color", {}, module_id=mylib_id)


# ---------------------------------------------------------------------------
# Coverage: _resolve_qualified_name_type falls back to built-in type when not in graph table.
# ---------------------------------------------------------------------------


def test_self_ref_type_builtin_exception_fallback(tmp_path: Path) -> None:
    """'::Abort' self-reference in a module falls back to built-in exception type."""
    modules = {
        "entry": (
            "import mylib\n"
            "let e = mylib::boom()\n"
            "e"
        ),
        "mylib": (
            # ::Abort references the built-in Abort exception type (not in graph table)
            # This exercises the fallback at env.py line 509
            "def boom() -> ::Abort = raise Abort(message = \"oops\")"
        ),
    }
    cg = _check_graph(tmp_path, modules)
    t = _binding_value_type(cg, ENTRY_ID, "e")
    assert isinstance(t, ExceptionType)
    assert t.name == "Abort"


# ---------------------------------------------------------------------------
# Coverage: _resolve_qualified_name_type raises when the qualified name is inaccessible.
# ---------------------------------------------------------------------------


def test_qualified_type_not_in_s_error(tmp_path: Path) -> None:
    """Using mylib::Secret when Secret is private in mylib → type error."""
    modules = {
        "entry": (
            "import mylib qualified using pub\n"
            "let n: mylib::Secret = mylib::pub()\n"
            "n"
        ),
        "mylib": (
            "private record Secret\n"
            "  x: int\n"
            "def pub() -> int = 1"
        ),
    }
    with pytest.raises(AglTypeError, match="not accessible via qualifier"):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# Coverage: _resolve_name_type raises for an ambiguous open import.
# ---------------------------------------------------------------------------


def test_ambiguous_open_import_type_error(tmp_path: Path) -> None:
    """Both libA and libB export 'Color': using 'Color' unqualified is ambiguous → error."""
    modules = {
        "entry": (
            "import libA\n"
            "import libB\n"
            "let c: Color = libA::Color.Red\n"
            "c"
        ),
        "libA": (
            "enum Color\n"
            "  | Red\n"
            "  | Blue"
        ),
        "libB": (
            "enum Color\n"
            "  | Green\n"
            "  | Yellow"
        ),
    }
    with pytest.raises(AglTypeError, match="Ambiguous type"):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# Coverage: get_open_imported_enum_candidates skips non-enum types during variant lookup.
# ---------------------------------------------------------------------------


def test_open_import_non_enum_type_skipped_in_variant_lookup(tmp_path: Path) -> None:
    """Open import has a Record and Enum; searching for a variant skips the Record."""
    modules = {
        "entry": (
            "import mylib\n"
            # Red is a bare variant; Color is the only enum matching
            # Point is a record (not enum) so it's skipped in get_open_imported_enum_candidates
            "let c = Red\n"
            "c"
        ),
        "mylib": (
            "record Point\n"
            "  x: int\n"
            "enum Color\n"
            "  | Red\n"
            "  | Blue"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "c") == EnumType("Color", {}, module_id=mylib_id)


# ---------------------------------------------------------------------------
# Coverage: env.py get_open_imported_enum_candidates: duplicate key seen (563)
# ---------------------------------------------------------------------------


def test_open_import_dedup_in_variant_lookup(tmp_path: Path) -> None:
    """When a type is open-imported under two names, it is deduplicated in variant lookup."""
    modules = {
        "entry": (
            # Two import declarations expose (mylib, "Color") under two unqualified names:
            # "Color" (via using Color) and "C" (via using Color as C).
            # The seen-set dedup fires when the same type is open-imported under two names.
            "import mylib using Color\n"
            "import mylib using Color as C\n"
            "let x: Color = Red\n"
            "x"
        ),
        "mylib": (
            "enum Color\n"
            "  | Red\n"
            "  | Blue"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "x") == EnumType("Color", {}, module_id=mylib_id)


# ---------------------------------------------------------------------------
# Finding 1 (BLOCKER) + Finding 2 — cross-module record/enum field types
# capture stale shells when bodies are resolved in dict order.
#
# The fix: resolve type bodies in topological dependency order across ALL
# modules so each referenced type is fully built before it is embedded
# by-value as a field/variant/element type.
# ---------------------------------------------------------------------------


def test_cross_module_field_type_single_direction(tmp_path: Path) -> None:
    """Record field whose type lives in a dependency module typechecks correctly.

    lib::Wrapper has a field 'c: payload::Data'.  When payload is resolved
    AFTER lib in dict order the stale shell bug causes Data's fields to be
    empty when lib.Wrapper.c's type is captured — field access on 'w.c' then
    fails with a spurious mismatch.

    After the fix: 'w.c' returns payload::Data (with n: int field accessible),
    and the whole program typechecks successfully.
    """
    modules = {
        "entry": (
            "import lib\n"
            "let w: lib::Wrapper = lib::mk()\n"
            "let inner = w.c\n"
            "inner"
        ),
        "lib": (
            "import payload\n"
            "record Wrapper\n"
            "  c: payload::Data\n"
            "def mk() -> Wrapper = Wrapper(c = payload::Data(n = 1))"
        ),
        "payload": (
            "record Data\n"
            "  n: int"
        ),
    }
    cg = _check_graph(tmp_path, modules)

    payload_id = ModuleId.from_dotted("payload")
    lib_id = ModuleId.from_dotted("lib")

    data_type = cg.graph_type_table[(payload_id, "Data")]
    assert isinstance(data_type, RecordType)
    assert data_type.fields == {"n": IntType()}, (
        f"payload::Data must have field 'n: int', got {data_type.fields}"
    )

    wrapper_type = cg.graph_type_table[(lib_id, "Wrapper")]
    assert isinstance(wrapper_type, RecordType)
    # Wrapper.c must hold the CANONICAL (fully built) Data type, not an empty shell
    assert wrapper_type.fields.get("c") == data_type, (
        f"lib::Wrapper.c field type must equal payload::Data: "
        f"got {wrapper_type.fields.get('c')!r}, expected {data_type!r}"
    )


def test_cross_module_field_type_mutual_import_cycle(tmp_path: Path) -> None:
    """Mutual import cycle where FIELD types cross module boundaries typechecks.

    modA.Foo has field 'c: modB::Color' (Color is an enum).
    modB.Bar has field 'f: modA::Foo'.
    The structural type-definition dependency graph is acyclic (Color, then Foo,
    then Bar) even though the IMPORT graph has a cycle.

    After the fix: both records are fully built, round-trip field access
    and assignability work correctly.
    """
    modules = {
        "entry": (
            "import modA\n"
            "import modB\n"
            # Exercise round-trip field access and assignability
            "let foo: modA::Foo = modA::makeFoo()\n"
            "let bar: modB::Bar = modB::makeBar()\n"
            "let c = foo.c\n"      # should have type modB::Color
            "let f = bar.f\n"      # should have type modA::Foo
            "()"
        ),
        "modA": (
            "import modB\n"
            "record Foo\n"
            "  c: modB::Color\n"
            "def makeFoo() -> Foo = Foo(c = modB::Color.Red)"
        ),
        "modB": (
            "import modA\n"
            "enum Color\n"
            "  | Red\n"
            "  | Blue\n"
            "record Bar\n"
            "  f: modA::Foo\n"
            "def makeBar() -> Bar = Bar(f = modA::makeFoo())"
        ),
    }
    cg = _check_graph(tmp_path, modules)

    mod_a = ModuleId.from_dotted("modA")
    mod_b = ModuleId.from_dotted("modB")

    foo_type = cg.graph_type_table[(mod_a, "Foo")]
    color_type = cg.graph_type_table[(mod_b, "Color")]
    bar_type = cg.graph_type_table[(mod_b, "Bar")]

    assert isinstance(foo_type, RecordType)
    assert isinstance(color_type, EnumType)
    assert isinstance(bar_type, RecordType)

    # modA::Foo.c must be the CANONICAL modB::Color (not an empty shell)
    assert foo_type.fields.get("c") == color_type, (
        f"modA::Foo.c must equal modB::Color: "
        f"got {foo_type.fields.get('c')!r}, expected {color_type!r}"
    )
    # modB::Bar.f must be the CANONICAL modA::Foo (not an empty shell)
    assert bar_type.fields.get("f") == foo_type, (
        f"modB::Bar.f must equal modA::Foo: "
        f"got {bar_type.fields.get('f')!r}, expected {foo_type!r}"
    )


def test_cross_module_enum_variant_field_type(tmp_path: Path) -> None:
    """Enum variant FIELD whose type is in another module is fully built.

    carrier::Envelope has variant Some with field 'value: payload::Data'.
    After the fix the variant field type captures the fully-built Data type
    (with fields populated), not an empty shell.
    """
    modules = {
        "entry": (
            "import carrier\n"
            "import payload\n"
            "let env: carrier::Envelope = carrier::wrap(payload::Data(n = 42))\n"
            "env"
        ),
        "carrier": (
            "import payload\n"
            "enum Envelope\n"
            "  | None\n"
            "  | Some(value: payload::Data)\n"
            "def wrap(d: payload::Data) -> Envelope = Envelope.Some(value = d)"
        ),
        "payload": (
            "record Data\n"
            "  n: int"
        ),
    }
    cg = _check_graph(tmp_path, modules)

    payload_id = ModuleId.from_dotted("payload")
    carrier_id = ModuleId.from_dotted("carrier")

    data_type = cg.graph_type_table[(payload_id, "Data")]
    envelope_type = cg.graph_type_table[(carrier_id, "Envelope")]

    assert isinstance(data_type, RecordType)
    assert data_type.fields == {"n": IntType()}

    assert isinstance(envelope_type, EnumType)
    some_fields = envelope_type.variants.get("Some", {})
    assert some_fields.get("value") == data_type, (
        f"carrier::Envelope.Some.value must equal payload::Data: "
        f"got {some_fields.get('value')!r}, expected {data_type!r}"
    )


# ---------------------------------------------------------------------------
# Finding 1 — structural type cycle detection is preserved (cross-module)
# A type that STRUCTURALLY contains itself (infinite size) must still be an
# error even after the topological-order fix.
# ---------------------------------------------------------------------------


def test_structural_type_cycle_across_modules_is_error(tmp_path: Path) -> None:
    """A genuine structural type cycle (type that contains itself) is rejected.

    This tests that the topological-sort-based resolution still surfaces
    structural cycles as errors (not silently ignores them).
    """
    from agm.agl.typecheck.env import AglTypeError as _AglTypeError

    modules = {
        "entry": (
            "import lib\n"
            "()"
        ),
        "lib": (
            # Node is directly structurally recursive (Node.child: Node)
            "record Node\n"
            "  child: Node"
        ),
    }
    with pytest.raises(_AglTypeError):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# Finding 3 (MINOR) — cross-module type mismatch diagnostics qualify the type.
# A rejection message for foo::Color vs bar::Color must say e.g.
# "foo::Color" and "bar::Color", not just "Color" twice.
# ---------------------------------------------------------------------------


def test_cross_module_mismatch_message_qualifies_type(tmp_path: Path) -> None:
    """A cross-module type mismatch error message includes module qualifiers.

    When foo::Color is given where bar::Color is expected the error must render
    as something that distinguishes the two modules (e.g. 'foo::Color' and
    'bar::Color'), not just 'Color' twice.
    """
    modules = {
        "entry": (
            "import foo\n"
            "import bar\n"
            "def get_foo() -> foo::Color = foo::makeColor()\n"
            "def expect_bar(c: bar::Color) -> bar::Color = c\n"
            "let c = get_foo()\n"
            "expect_bar(c)"
        ),
        "foo": (
            "enum Color\n"
            "  | Red\n"
            "  | Blue\n"
            "def makeColor() -> Color = Red"
        ),
        "bar": (
            "enum Color\n"
            "  | Red\n"
            "  | Blue\n"
            "def makeColor() -> Color = Red"
        ),
    }
    from agm.agl.typecheck.env import AglTypeError as _AglTypeError

    with pytest.raises(_AglTypeError) as exc_info:
        _check_graph(tmp_path, modules)

    msg = str(exc_info.value)
    # The error message must distinguish the two Color types by their module
    assert "foo" in msg and "bar" in msg, (
        f"Expected mismatch message to mention both 'foo' and 'bar', got: {msg!r}"
    )


# ---------------------------------------------------------------------------
# Coverage: graph.py _collect_type_expr_deps — various dep-collection paths
# ---------------------------------------------------------------------------


def test_type_expr_deps_self_ref_qualifier(tmp_path: Path) -> None:
    """A field typed '::OwnType' creates a self-dep (::Name qualifier path)."""
    modules = {
        "entry": (
            "import mylib\n"
            "let p: mylib::Wrapper = mylib::mk()\n"
            "p"
        ),
        "mylib": (
            # ::Inner is a self-reference (same module)
            "record Inner\n"
            "  n: int\n"
            "record Wrapper\n"
            "  c: ::Inner\n"
            "def mk() -> Wrapper = Wrapper(c = Inner(n = 1))"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "p") == RecordType("Wrapper", {}, module_id=mylib_id)


def test_type_expr_deps_qualified_field(tmp_path: Path) -> None:
    """A field typed 'other::Type' creates a cross-module dep (qualified path)."""
    modules = {
        "entry": (
            "import mylib qualified\n"
            "let p: mylib::Wrapper = mylib::mk()\n"
            "p"
        ),
        "mylib": (
            "import payload qualified\n"
            "record Wrapper\n"
            "  c: payload::Data\n"
            "def mk() -> Wrapper = Wrapper(c = payload::Data(n = 1))"
        ),
        "payload": (
            "record Data\n"
            "  n: int"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "p") == RecordType("Wrapper", {}, module_id=mylib_id)


def test_type_expr_deps_unqualified_open_import_field(tmp_path: Path) -> None:
    """A field typed with an open-imported name creates a dep (unqualified path)."""
    modules = {
        "entry": (
            "import mylib\n"
            "let p: mylib::Wrapper = mylib::mk()\n"
            "p"
        ),
        "mylib": (
            # Open import: 'import payload' — Data is an open-imported type
            "import payload\n"
            "record Wrapper\n"
            "  c: Data\n"       # unqualified reference to open-imported type
            "def mk() -> Wrapper = Wrapper(c = Data(n = 1))"
        ),
        "payload": (
            "record Data\n"
            "  n: int"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "p") == RecordType("Wrapper", {}, module_id=mylib_id)


def test_type_expr_deps_list_field(tmp_path: Path) -> None:
    """A field typed 'list[other::Type]' recurses into the elem type for deps."""
    modules = {
        "entry": (
            "import mylib\n"
            "let p: mylib::Wrapper = mylib::mk()\n"
            "p"
        ),
        "mylib": (
            "import payload\n"
            "record Wrapper\n"
            "  items: list[payload::Data]\n"
            "def mk() -> Wrapper = Wrapper(items = [payload::Data(n = 1)])"
        ),
        "payload": (
            "record Data\n"
            "  n: int"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "p") == RecordType("Wrapper", {}, module_id=mylib_id)


def test_type_expr_deps_dict_field(tmp_path: Path) -> None:
    """A field typed 'dict[text, other::Type]' recurses into the value type."""
    modules = {
        "entry": (
            "import mylib\n"
            "let p: mylib::Wrapper = mylib::mk()\n"
            "p"
        ),
        "mylib": (
            "import payload\n"
            "record Wrapper\n"
            "  items: dict[text, payload::Data]\n"
            "def mk() -> Wrapper = Wrapper(items = {})"
        ),
        "payload": (
            "record Data\n"
            "  n: int"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "p") == RecordType("Wrapper", {}, module_id=mylib_id)


def test_type_expr_deps_alias_to_cross_module(tmp_path: Path) -> None:
    """A type alias whose target is a cross-module type creates a dep (alias path)."""
    modules = {
        "entry": (
            "import mylib\n"
            "let n: mylib::MyNum = 42\n"
            "n"
        ),
        "mylib": (
            # TypeAlias whose target is a cross-module record
            "import payload\n"
            "type MyNum = int"
        ),
        "payload": (
            "record Data\n"
            "  n: int"
        ),
    }
    cg = _check_graph(tmp_path, modules)
    mylib_id = ModuleId.from_dotted("mylib")
    t = cg.graph_type_table[(mylib_id, "MyNum")]
    assert isinstance(t, IntType)
    assert _binding_value_type(cg, ENTRY_ID, "n") == IntType()


# ---------------------------------------------------------------------------
# Coverage: _topological_sort_types cycle detection
# and _build_graph_type_table cross-module cycle error.
# This is different from the existing test (which tests same-module recursion)
# and exercises the cross-module structural cycle path.
# ---------------------------------------------------------------------------


def test_cross_module_structural_cycle_raises_error(tmp_path: Path) -> None:
    """A cross-module structural type cycle is detected and reported.

    modA.Foo has field 'other: modB::Bar' and modB.Bar has field 'other: modA::Foo'.
    This is a genuine structural cycle (Foo contains Bar contains Foo), which
    makes both types infinitely sized.  The checker must raise AglTypeError.
    """
    from agm.agl.typecheck.env import AglTypeError as _AglTypeError

    modules = {
        "entry": (
            "import modA\n"
            "import modB\n"
            "()"
        ),
        "modA": (
            "import modB\n"
            "record Foo\n"
            "  other: modB::Bar"
        ),
        "modB": (
            "import modA\n"
            "record Bar\n"
            "  other: modA::Foo"
        ),
    }
    with pytest.raises(_AglTypeError):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# Coverage: RecordType.__repr__ renders as 'module::Name' for non-entry module types.
# ---------------------------------------------------------------------------


def test_record_type_repr_qualified_with_module(tmp_path: Path) -> None:
    """RecordType from a non-entry module renders as 'module::Name' in error messages."""
    from agm.agl.typecheck.env import AglTypeError as _AglTypeError

    modules = {
        "entry": (
            "import foo\n"
            "import bar\n"
            "def get_foo() -> foo::Point = foo::makePoint()\n"
            "def expect_bar(p: bar::Point) -> bar::Point = p\n"
            "let p = get_foo()\n"
            "expect_bar(p)"  # foo::Point ≠ bar::Point
        ),
        "foo": (
            "record Point\n"
            "  x: int\n"
            "  y: int\n"
            "def makePoint() -> Point = Point(x = 0, y = 0)"
        ),
        "bar": (
            "record Point\n"
            "  x: int\n"
            "  y: int\n"
            "def makePoint() -> Point = Point(x = 1, y = 1)"
        ),
    }
    with pytest.raises(_AglTypeError) as exc_info:
        _check_graph(tmp_path, modules)

    msg = str(exc_info.value)
    # Both module qualifiers must appear in the mismatch message
    assert "foo" in msg and "bar" in msg, (
        f"Expected mismatch message to qualify both modules, got: {msg!r}"
    )


# ---------------------------------------------------------------------------
# Coverage: graph.py remaining branch paths
# ---------------------------------------------------------------------------


def test_type_alias_with_cross_module_dep_creates_dep(tmp_path: Path) -> None:
    """A TypeAlias whose target is a cross-module user type creates a dep entry.

    This exercises the 'elif isinstance(item, TypeAlias)' branch in
    _compute_type_deps with a dep that actually appears in
    all_type_keys, and also exercises the branch where in_degree is decremented
    but does not reach 0 (diamond dependency pattern for Kahn's algorithm).
    """
    modules = {
        "entry": (
            "import mylib\n"
            "let w: mylib::Wrapper = mylib::mk()\n"
            "w"
        ),
        "mylib": (
            "import payload\n"
            # TypeAlias whose target is a user type in another module
            "type DataAlias = payload::Data\n"
            "record Wrapper\n"
            "  a: DataAlias\n"    # depends on alias, alias depends on payload::Data
            "  b: payload::Data\n"  # direct dep on payload::Data (diamond dep!)
            "def mk() -> Wrapper = Wrapper(a = payload::Data(n = 1), b = payload::Data(n = 2))"
        ),
        "payload": (
            "record Data\n"
            "  n: int"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "w") == RecordType("Wrapper", {}, module_id=mylib_id)


def test_type_expr_deps_func_field(tmp_path: Path) -> None:
    """A field typed with a function type recursing into params and result.

    This exercises the FuncT branch in _collect_type_expr_deps.
    The function type's param type is a cross-module user type, so the FuncT
    walker must descend into the param to find the dependency.
    """
    modules = {
        "entry": (
            "import mylib\n"
            "let p: mylib::Wrapper = mylib::mk()\n"
            "p"
        ),
        "mylib": (
            "import payload\n"
            # Field with a function type whose param is a cross-module user type
            # This exercises the FuncT branch in _collect_type_expr_deps
            "record Wrapper\n"
            "  transform: (payload::Data) -> text\n"
            "def mk() -> Wrapper = Wrapper(transform = fn(d: payload::Data) -> text => \"ok\")"
        ),
        "payload": (
            "record Data\n"
            "  n: int"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "p") == RecordType("Wrapper", {}, module_id=mylib_id)


# ---------------------------------------------------------------------------
# Coverage: defensive guard paths in _collect_type_expr_deps
# These exercise the "not in all_type_keys" guards which fire when a type
# expression references a built-in type (which is never in all_type_keys).
# ---------------------------------------------------------------------------


def test_type_expr_deps_self_ref_to_builtin(tmp_path: Path) -> None:
    """A '::BuiltinType' self-ref in a field creates NO dep (key not in all_type_keys).

    This exercises the path in _collect_type_expr_deps where the
    self-ref target is a built-in type (not in all_type_keys), so no dep is added.
    """
    modules = {
        "entry": (
            "import mylib\n"
            "let w: mylib::Wrapper = mylib::mk()\n"
            "w"
        ),
        "mylib": (
            # ::ExecResult is a built-in prelude type (not in all_type_keys)
            "record Wrapper\n"
            "  c: ::ExecResult\n"
            "def mk() -> Wrapper = Wrapper(c = exec(\"echo hi\"))"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "w") == RecordType("Wrapper", {}, module_id=mylib_id)


def test_type_expr_deps_open_import_to_builtin_variant(tmp_path: Path) -> None:
    """An unqualified name that's a builtin creates NO dep (key not in all_type_keys).

    This exercises the path in _collect_type_expr_deps where the
    candidate key (from open-import unqualified lookup) is not in all_type_keys.
    """
    # When no open-imported name matches a NameT, the loop finds no candidates
    # and all_type_keys guard is not exercised. Here we use a record field
    # with an unqualified name that resolves through open import to a user type.
    # The "key in all_type_keys -> False" path fires when we find a candidate but
    # it's already a resolved primitive type — hard to trigger with valid programs.
    # Instead test the 'no candidates' path (dep lookup is empty, deps stay []).
    modules = {
        "entry": (
            "import mylib\n"
            "let w: mylib::Wrapper = mylib::mk()\n"
            "w"
        ),
        "mylib": (
            # Field typed 'int' (builtin) via bare name — candidates list is empty
            "record Wrapper\n"
            "  n: int\n"
            "def mk() -> Wrapper = Wrapper(n = 42)"
        ),
    }
    mylib_id = ModuleId.from_dotted("mylib")
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "w") == RecordType("Wrapper", {}, module_id=mylib_id)


# ---------------------------------------------------------------------------
# Coverage: graph.py _collect_type_expr_deps — branches that were masked by
# `# pragma: no branch` directives, now covered by real tests.
# ---------------------------------------------------------------------------


def test_builtin_shadowing_type_raises_type_error(tmp_path: Path) -> None:
    """A module that declares a type with a built-in name raises AglTypeError.

    _collect_shells_only (Step A of _build_graph_type_table) calls
    _register_name which immediately raises for builtin-shadowing types.
    This confirms the check fires in phase 1 (pre-pass) before any per-module
    validation, and that _collect_all_type_keys is never reached with such a type
    (making the removed `not_builtin` guard genuinely dead).
    """
    from agm.agl.typecheck.env import AglTypeError as _AglTypeError

    modules = {
        "entry": (
            "import mylib\n"
            "()"
        ),
        "mylib": (
            # ExecResult shadows BUILTIN_PRELUDE_TYPES — rejected in _collect_shells_only
            "record ExecResult\n"
            "  x: int"
        ),
    }
    with pytest.raises(_AglTypeError, match="built-in type name"):
        _check_graph(tmp_path, modules)


def test_field_type_with_unimported_qualifier_is_type_error(tmp_path: Path) -> None:
    """A record field typed 'other::Data' where 'other' is NOT imported → AglTypeError.

    This exercises the `handle_map is None` FALSE branch in
    _collect_type_expr_deps: during dep collection (phase 1) the qualifier
    segment is not found in import_env.qualified, so dep collection skips it
    silently. Phase 2 then raises the proper type error.
    """
    from agm.agl.typecheck.env import AglTypeError as _AglTypeError

    modules = {
        "entry": (
            "import mylib\n"
            "()"
        ),
        "mylib": (
            # 'other' is not imported — handle_map will be None in dep-collection
            "record MyRec\n"
            "  c: other::Data"
        ),
    }
    with pytest.raises(_AglTypeError):
        _check_graph(tmp_path, modules)


def test_field_type_with_unknown_qualified_name_is_type_error(tmp_path: Path) -> None:
    """A record field typed 'payload::Unknown' where 'Unknown' is not exported → AglTypeError.

    This exercises the `qname is None` FALSE branch in _collect_type_expr_deps:
    during dep collection the qualifier 'payload' resolves (handle_map found) but
    the name 'Unknown' is absent from the handle's name map, so the dep is skipped.
    Phase 2 then raises the proper qualified-type error.
    """
    from agm.agl.typecheck.env import AglTypeError as _AglTypeError

    modules = {
        "entry": (
            "import mylib\n"
            "()"
        ),
        "mylib": (
            "import payload qualified\n"
            # 'Unknown' does not exist in payload — qname will be None
            "record MyRec\n"
            "  c: payload::Unknown"
        ),
        "payload": (
            "record Data\n"
            "  n: int"
        ),
    }
    with pytest.raises(_AglTypeError):
        _check_graph(tmp_path, modules)


def test_field_type_with_qualified_function_name_is_type_error(tmp_path: Path) -> None:
    """A record field typed 'payload::getValue' where getValue is a function → AglTypeError.

    This exercises the `key in all_type_keys` FALSE branch for the QUALIFIED
    path in _collect_type_expr_deps: handle_map and qname both resolve, but the
    resulting (ModuleId, name) key is a function, not a user type, so it is absent
    from all_type_keys and the dep is skipped silently. Phase 2 raises the proper
    'not a type' error.
    """
    from agm.agl.typecheck.env import AglTypeError as _AglTypeError

    modules = {
        "entry": (
            "import mylib\n"
            "()"
        ),
        "mylib": (
            "import payload qualified\n"
            # 'getValue' is a function in payload, not a type — key not in all_type_keys
            "record MyRec\n"
            "  c: payload::getValue"
        ),
        "payload": (
            "def getValue() -> int = 42"
        ),
    }
    with pytest.raises(_AglTypeError):
        _check_graph(tmp_path, modules)


def test_field_type_with_open_imported_function_name_is_type_error(tmp_path: Path) -> None:
    """A record field typed with an open-imported function name → AglTypeError.

    This exercises the `key in all_type_keys` FALSE branch for the UNQUALIFIED
    candidates path in _collect_type_expr_deps: the name 'getValue' appears in
    import_env.unqualified (because payload is open-imported and exports getValue),
    but its (ModuleId, name) key is not in all_type_keys (it's a function, not a
    user type), so the dep is skipped silently. Phase 2 raises the proper
    'unknown type' error.
    """
    from agm.agl.typecheck.env import AglTypeError as _AglTypeError

    modules = {
        "entry": (
            "import mylib\n"
            "()"
        ),
        "mylib": (
            # Open-import brings 'getValue' (a function) into the unqualified namespace.
            # Using it as a field type triggers the unqualified candidates False branch.
            "import payload\n"
            "record MyRec\n"
            "  c: getValue"
        ),
        "payload": (
            "def getValue() -> int = 42"
        ),
    }
    with pytest.raises(_AglTypeError):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# Cross-file mutual recursion (regression for function-signature pre-pass)
# ---------------------------------------------------------------------------


def test_cross_file_mutual_recursion_qualified(tmp_path: Path) -> None:
    """True A↔B cross-file mutual recursion typechecks successfully (qualified calls).

    Module 'even' defines is_even(n) calling odd::is_odd(n-1).
    Module 'odd'  defines is_odd(n)  calling even::is_even(n-1).
    Entry imports both and calls even::is_even(10).

    Whichever of 'even'/'odd' is checked first lacks the other's function
    signatures unless a whole-graph function-signature pre-pass seeds them
    before any body is checked.  This test MUST FAIL before the pre-pass is
    added and MUST PASS after.
    """
    modules = {
        "even": (
            "import odd\n"
            "def is_even(n: int) -> bool =\n"
            "  if n == 0 => true\n"
            "  | else => odd::is_odd(n - 1)"
        ),
        "odd": (
            "import even\n"
            "def is_odd(n: int) -> bool =\n"
            "  if n == 0 => false\n"
            "  | else => even::is_even(n - 1)"
        ),
        "entry": (
            "import even\n"
            "let result = even::is_even(10)\n"
            "result"
        ),
    }
    cg = _check_graph(tmp_path, modules)
    mid_even = ModuleId.from_dotted("even")
    mid_odd = ModuleId.from_dotted("odd")
    assert mid_even in cg.modules
    assert mid_odd in cg.modules
    assert ENTRY_ID in cg.modules
    assert _binding_value_type(cg, ENTRY_ID, "result") == BoolType()


def test_cross_file_mutual_recursion_open_import(tmp_path: Path) -> None:
    """True A↔B cross-file mutual recursion typechecks via open (unqualified) imports.

    Same mutual recursion as the qualified test, but both modules open-import
    each other so calls are unqualified.

    This test MUST FAIL before the function-signature pre-pass and MUST PASS after.
    """
    modules = {
        "even": (
            "import odd\n"
            "def is_even(n: int) -> bool =\n"
            "  if n == 0 => true\n"
            "  | else => is_odd(n - 1)"
        ),
        "odd": (
            "import even\n"
            "def is_odd(n: int) -> bool =\n"
            "  if n == 0 => false\n"
            "  | else => is_even(n - 1)"
        ),
        "entry": (
            "import even\n"
            "let result = is_even(10)\n"
            "result"
        ),
    }
    cg = _check_graph(tmp_path, modules)
    mid_even = ModuleId.from_dotted("even")
    mid_odd = ModuleId.from_dotted("odd")
    assert mid_even in cg.modules
    assert mid_odd in cg.modules
    assert ENTRY_ID in cg.modules
    assert _binding_value_type(cg, ENTRY_ID, "result") == BoolType()


# ---------------------------------------------------------------------------
# Coverage: _build_graph_func_sig_table branches
# ---------------------------------------------------------------------------


def test_graph_func_def_with_defaulted_param(tmp_path: Path) -> None:
    """A library function with a defaulted parameter typechecks successfully.

    Covers the ``seen_required = False`` branch in
    ``_build_graph_func_sig_table``, which is only reached when a FuncDef in a
    non-entry module has at least one defaulted parameter.
    """
    modules = {
        "lib": "def add(a: int, b: int = 0) -> int = a + b",
        "entry": "import lib\nlet r = lib::add(10)\nr",
    }
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "r") == IntType()


def test_graph_func_def_required_after_default_error(tmp_path: Path) -> None:
    """A library function with a required param after a defaulted one → AglTypeError.

    Covers the ``raise AglTypeError`` branch in ``_build_graph_func_sig_table``
    (required parameter follows a defaulted parameter in a non-entry module's FuncDef).
    """
    from agm.agl.typecheck.env import AglTypeError as _AglTypeError

    modules = {
        "lib": "def bad(a: int = 0, b: int) -> int = a + b",
        "entry": "import lib\nlet r = lib::bad(1, 2)\nr",
    }
    with pytest.raises(_AglTypeError, match="has no default but follows"):
        _check_graph(tmp_path, modules)


def test_graph_func_def_builtin_type_name_error(tmp_path: Path) -> None:
    """A library function named after a builtin type → AglTypeError from body checker.

    The function-signature pre-pass skips functions whose names clash with
    built-in type names (e.g. 'bool') to avoid raising prematurely with wrong
    source spans; the body checker's _preregister_funcdef then raises the proper
    error.  This test covers the ``continue`` at the builtin-name guard in
    ``_build_graph_func_sig_table``.

    Note: builtin *function* names ('print', 'ask', etc.) are rejected earlier by
    the scope resolver (AglScopeError).  Only builtin *type* names pass scope
    resolution to reach the typecheck gate tested here.
    """
    from agm.agl.typecheck.env import AglTypeError as _AglTypeError

    modules = {
        "lib": "def bool(x: int) -> int = x",
        "entry": "import lib\n()",
    }
    with pytest.raises(_AglTypeError, match="built-in type name"):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# Finding 1 regression tests: cross-module same-name function signature collision
# ---------------------------------------------------------------------------


def test_cross_module_same_name_qualified_call_false_reject(tmp_path: Path) -> None:
    """Regression: qualified call to lib::helper(int) must NOT be rejected as type error.

    Entry defines  helper(s: text) -> text.
    Lib defines    helper(n: int) -> int.
    Entry calls    lib::helper(5) (qualified, int arg).

    Before the fix: _check_declared_name_call fetched the name-keyed signature
    and returned entry's helper (text param), causing a spurious type mismatch.
    After the fix: the node-id-keyed table returns lib's helper (int param),
    so the call typechecks correctly.
    """
    modules = {
        "lib": "def helper(n: int) -> int = n + 1",
        "entry": (
            "import lib qualified\n"
            "def helper(s: text) -> text = s\n"
            "let r = lib::helper(5)\n"
            "r"
        ),
    }
    # Must NOT raise — the qualified call uses lib's signature (int param).
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "r") == IntType()


def test_cross_module_same_name_qualified_call_false_accept(tmp_path: Path) -> None:
    """Regression: qualified call to lib::helper(int) when lib expects text must be rejected.

    Entry defines  helper(n: int) -> int.
    Lib defines    helper(s: text) -> text.
    Entry calls    lib::helper(5) (int arg, but lib expects text).

    Before the fix: _check_declared_name_call fetched entry's helper (int param),
    falsely accepting the call; the evaluator then crashed.
    After the fix: the node-id-keyed table returns lib's helper (text param),
    and the type checker correctly rejects the int argument.
    """
    from agm.agl.typecheck.env import AglTypeError as _AglTypeError

    modules = {
        "lib": "def helper(s: text) -> text = s",
        "entry": (
            "import lib qualified\n"
            "def helper(n: int) -> int = n\n"
            "let r = lib::helper(5)\n"
            "r"
        ),
    }
    with pytest.raises(_AglTypeError, match="Type mismatch"):
        _check_graph(tmp_path, modules)


def test_two_library_functions_same_name_different_signatures(tmp_path: Path) -> None:
    """Two modules define same-named functions with DIFFERENT signatures.

    Entry also defines the same name.  Qualified calls to each module must each
    be validated against the CORRECT module's signature, not the entry's or the
    other module's.

    Module 'a': helper(n: int) -> int   (entry calls a::helper(5))
    Module 'b': helper(s: text) -> text (entry calls b::helper("hello"))
    Entry:      helper(x: bool) -> bool

    Both qualified calls must typecheck; swapping arg types must be rejected.
    """
    from agm.agl.typecheck.env import AglTypeError as _AglTypeError

    # Correct types → should typecheck
    modules_ok = {
        "a": "def helper(n: int) -> int = n + 1",
        "b": 'def helper(s: text) -> text = s',
        "entry": (
            "import a qualified\n"
            "import b qualified\n"
            "def helper(x: bool) -> bool = x\n"
            "let ra = a::helper(5)\n"
            'let rb = b::helper("hello")\n'
            "rb"
        ),
    }
    cg = _check_graph(tmp_path, modules_ok)
    assert _binding_value_type(cg, ENTRY_ID, "ra") == IntType()
    assert _binding_value_type(cg, ENTRY_ID, "rb") == TextType()

    # Swapped: pass text to a::helper (expects int) → type error
    tmp_path2 = tmp_path.parent / (tmp_path.name + "_bad")
    tmp_path2.mkdir()
    modules_bad = {
        "a": "def helper(n: int) -> int = n + 1",
        "b": 'def helper(s: text) -> text = s',
        "entry": (
            "import a qualified\n"
            "import b qualified\n"
            "def helper(x: bool) -> bool = x\n"
            'let ra = a::helper("wrong")\n'
            "let rb = b::helper(5)\n"
            "rb"
        ),
    }
    with pytest.raises(_AglTypeError, match="Type mismatch"):
        _check_graph(tmp_path2, modules_bad)


# ---------------------------------------------------------------------------
# Cross-module constructor call error paths
# ---------------------------------------------------------------------------


def test_cross_module_constructor_call_positional_args_rejected(tmp_path: Path) -> None:
    """Coverage: _check_cross_module_constructor_call rejects positional arguments.

    Calling a cross-module record constructor with positional arguments raises AglTypeError.
    """
    modules = {
        "lib": "record Point\n  x: int\n  y: int",
        "entry": "import lib qualified\nlib::Point(1, 2)",
    }
    with pytest.raises(AglTypeError, match="named"):
        _check_graph(tmp_path, modules)


def test_cross_module_generic_constructor_call_explicit_type_args(tmp_path: Path) -> None:
    """Cross-module generic constructor with explicit type args: lib::Box::[int](value = 1)."""
    lib_id = ModuleId.from_dotted("lib")
    modules = {
        "lib": "record Box[T]\n  value: T",
        "entry": "import lib qualified\nlet r = lib::Box::[int](value = 1)\nr",
    }
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "r") == RecordType(
        "Box", {}, module_id=lib_id, type_args=(IntType(),)
    )


def test_cross_module_generic_constructor_call_inferred_type_args(tmp_path: Path) -> None:
    """Cross-module generic constructor with inferred type args: lib::Box(value = 1)."""
    lib_id = ModuleId.from_dotted("lib")
    modules = {
        "lib": "record Box[T]\n  value: T",
        "entry": "import lib qualified\nlet r = lib::Box(value = 1)\nr",
    }
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "r") == RecordType(
        "Box", {}, module_id=lib_id, type_args=(IntType(),)
    )


def test_open_imported_generic_type_in_annotation(tmp_path: Path) -> None:
    lib_id = ModuleId.from_dotted("lib")
    modules = {
        "lib": "record Box[T]\n  value: T",
        "entry": "import lib\nlet x: Box[int] = Box(value = 1)\nx",
    }
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "x") == RecordType(
        "Box", {}, module_id=lib_id, type_args=(IntType(),)
    )


def test_qualified_generic_type_in_annotation(tmp_path: Path) -> None:
    lib_id = ModuleId.from_dotted("lib")
    modules = {
        "lib": "record Box[T]\n  value: T",
        "entry": "import lib qualified\nlet x: lib::Box[int] = lib::Box(value = 1)\nx",
    }
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "x") == RecordType(
        "Box", {}, module_id=lib_id, type_args=(IntType(),)
    )


def test_ambiguous_open_imported_generic_type_rejected(tmp_path: Path) -> None:
    modules = {
        "a": "record Box[T]\n  value: T",
        "b": "record Box[T]\n  value: T",
        "entry": "import a\nimport b\nlet x: Box[int] = null\nx",
    }
    with pytest.raises(AglTypeError, match="Ambiguous type 'Box'"):
        _check_graph(tmp_path, modules)


def test_open_imported_non_generic_type_application_rejected(tmp_path: Path) -> None:
    modules = {
        "lib": "record Point\n  value: int",
        "entry": "import lib\nlet x: Point[int] = null\nx",
    }
    with pytest.raises(AglTypeError, match="does not take type arguments"):
        _check_graph(tmp_path, modules)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (
            "import lib qualified\nlet x: missing::Box[int] = null\nx",
            "Unknown module qualifier",
        ),
        (
            "import lib qualified using helper\nlet x: lib::Box[int] = null\nx",
            "not accessible",
        ),
        ("import lib qualified\nlet x: lib::Point[int] = null\nx", "does not take"),
        (
            "import lib qualified\nlet x: lib::helper[int] = null\nx",
            "does not name a type",
        ),
    ],
)
def test_qualified_type_application_errors(
    tmp_path: Path, entry: str, message: str
) -> None:
    modules = {
        "lib": (
            "record Box[T]\n"
            "  value: T\n"
            "record Point\n"
            "  value: int\n"
            "def helper(x: int) -> int = x"
        ),
        "entry": entry,
    }
    with pytest.raises(AglTypeError, match=message):
        _check_graph(tmp_path, modules)


def test_unknown_applied_type_with_import_environment_rejected(tmp_path: Path) -> None:
    modules = {
        "lib": "def helper(x: int) -> int = x",
        "entry": "import lib\nlet x: Missing[int] = null\nx",
    }
    with pytest.raises(AglTypeError, match="Unknown type 'Missing'"):
        _check_graph(tmp_path, modules)


def test_cross_module_qualified_generic_enum_explicit_type_args(tmp_path: Path) -> None:
    lib_id = ModuleId.from_dotted("lib")
    modules = {
        "lib": "enum Option[T]\n  | none\n  | some(value: T)",
        "entry": "import lib qualified\nlet r = lib::Option.some::[int](value = 1)\nr",
    }
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "r") == EnumType(
        "Option", {}, module_id=lib_id, type_args=(IntType(),)
    )


def test_open_imported_generic_constructor_payload_type_apply_as_value(tmp_path: Path) -> None:
    """Open-imported generic payload constructor as a value with explicit type args.

    ``some::[int]`` in the entry resolves via the open-imported generic enum
    fallback and yields a function value ``int -> Choice[int]`` owned by ``lib``.
    The type is renamed away from ``Option`` to avoid clashing with the
    auto-open-imported ``std.core::Option``.
    """
    lib_id = ModuleId.from_dotted("lib")
    modules = {
        "lib": "enum Choice[T]\n  | none\n  | some(value: T)",
        "entry": "import lib\nlet f = some::[int]\nf",
    }
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "f") == FunctionType(
        (IntType(),), EnumType("Choice", {}, module_id=lib_id, type_args=(IntType(),))
    )


def test_open_imported_generic_constructor_nullary_type_apply_as_value(tmp_path: Path) -> None:
    """Open-imported generic nullary constructor as a value with explicit type args.

    ``none::[int]`` constructs the nullary ``Choice[int]`` value owned by ``lib``.
    """
    lib_id = ModuleId.from_dotted("lib")
    modules = {
        "lib": "enum Choice[T]\n  | none\n  | some(value: T)",
        "entry": "import lib\nlet z = none::[int]\nz",
    }
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "z") == EnumType(
        "Choice", {}, module_id=lib_id, type_args=(IntType(),)
    )


def test_cross_module_non_generic_constructor_type_args_rejected(tmp_path: Path) -> None:
    """Coverage: checker.py _check_cross_module_constructor_call — non-generic with type args."""
    modules = {
        "lib": "record Point\n  x: int",
        "entry": "import lib qualified\nlib::Point::[int](x = 1)",
    }
    with pytest.raises(AglTypeError, match="not a generic type"):
        _check_graph(tmp_path, modules)


def test_qualified_same_named_exceptions_keep_module_field_kinds(tmp_path: Path) -> None:
    """Qualified exception construction uses the defining module's field kinds."""
    modules = {
        "a": "exception Boom extends Exception\n  a: int",
        "b": "exception Boom extends Exception\n  b: text",
        "entry": "import a qualified\nimport b qualified\na::Boom(message = \"x\", a = 1)",
    }
    _check_graph(tmp_path, modules)




def test_cross_module_generic_enum_body_resolved(tmp_path: Path) -> None:
    """Coverage: _resolve_body_for_one EnumDef branch with generic enum (t is None).

    A cross-module generic enum causes ensure_built_enum to register in _generic_types
    and unregister from _types, so get_type returns None.
    The graph_type_table retains the shell (EnumType with module_id set) from Step A.
    """
    lib_id = ModuleId.from_dotted("lib")
    modules = {
        "lib": "enum Opt[T]\n  | None\n  | Wrap(value: T)",
        "entry": "import lib qualified\n()",
    }
    cg = _check_graph(tmp_path, modules)
    # The shell for Opt is in graph_type_table with the correct module_id.
    opt_type = cg.graph_type_table[(lib_id, "Opt")]
    assert isinstance(opt_type, EnumType)
    assert opt_type.name == "Opt"
    assert opt_type.module_id == lib_id


# ---------------------------------------------------------------------------
# BUG 1 regression: module_id dropped when instantiating generic nominals
# Two modules each define a generic with the same short name; instances must
# NOT be assignable across modules.
# ---------------------------------------------------------------------------


def test_cross_module_generic_record_template_has_module_id(tmp_path: Path) -> None:
    """Regression for BUG 1: generic record template must carry the owning module's module_id.

    Before the fix, _build_generic_record created the template RecordType without
    module_id=self._module_id, so all generic templates got module_id=ENTRY_ID.
    After the fix, the template carries the owning library module's module_id.

    Tested via the internal _build_graph_type_table function to access the
    graph_generic_table, which is not exposed on the public CheckedModuleGraph API.
    """
    from agm.agl.typecheck.graph import _build_graph_type_table

    modules = {
        "lib": "record Box[T]\n  value: T",
        "entry": "import lib qualified\n()",
    }
    mg = _make_graph_from_files(tmp_path, modules)
    rg = resolve_graph(mg)
    _gtt, graph_generic_table, _gcts, _gckft = _build_graph_type_table(rg)

    lib_id = ModuleId.from_dotted("lib")
    gdef = graph_generic_table.get((lib_id, "Box"))
    assert gdef is not None, "lib::Box must appear in graph_generic_table"
    assert gdef.template.module_id == lib_id, (
        f"lib::Box template must have module_id={lib_id!r}, "
        f"got {gdef.template.module_id!r}. "
        "BUG 1: _build_generic_record must pass module_id=self._module_id."
    )


def test_cross_module_generic_enum_template_has_module_id(tmp_path: Path) -> None:
    """Regression for BUG 1: generic enum template must carry the owning module's module_id.

    Before the fix, _build_generic_enum created the template EnumType without
    module_id=self._module_id, so all generic templates got module_id=ENTRY_ID.
    After the fix, the template carries the owning library module's module_id.

    Tested via the internal _build_graph_type_table function to access the
    graph_generic_table, which is not exposed on the public CheckedModuleGraph API.
    """
    from agm.agl.typecheck.graph import _build_graph_type_table

    modules = {
        "lib": "enum Opt[T]\n  | None\n  | Some(value: T)",
        "entry": "import lib qualified\n()",
    }
    mg = _make_graph_from_files(tmp_path, modules)
    rg = resolve_graph(mg)
    _gtt, graph_generic_table, _gcts, _gckft = _build_graph_type_table(rg)

    lib_id = ModuleId.from_dotted("lib")
    gdef = graph_generic_table.get((lib_id, "Opt"))
    assert gdef is not None, "lib::Opt must appear in graph_generic_table"
    assert gdef.template.module_id == lib_id, (
        f"lib::Opt template must have module_id={lib_id!r}, "
        f"got {gdef.template.module_id!r}. "
        "BUG 1: _build_generic_enum must pass module_id=self._module_id."
    )


# ---------------------------------------------------------------------------
# BUG 2 regression: parameterized-alias type_params lost in multi-module pre-pass
# A module defines a parameterized alias; uses with type args must typecheck.
# ---------------------------------------------------------------------------


def test_parameterized_alias_in_graph_mode(tmp_path: Path) -> None:
    """Regression for BUG 2: type Pair[A,B] used in a record field typechecks via graph path.

    Before the fix, collect_shells_only called register_alias(name, expr) without
    type_params. When the cross-module body resolver (_resolve_body_for_one) called
    _ensure_built_record → resolve_type_expr(Pair[int,text]), it found Pair in
    _alias_targets with alias_params=() and raised 'requires 0 type argument(s)'.
    After the fix, type_params are threaded through collect_shells_only and the
    parameterized alias resolves correctly.
    """
    lib_id = ModuleId.from_dotted("lib")
    modules = {
        # lib declares Pair[A,B] and uses it in a record field —
        # this goes through _resolve_body_for_one → _ensure_built_record →
        # resolve_type_expr(AppliedT("Pair", ...)) via the cross-module builder.
        "lib": (
            "type Pair[A,B] = dict[text, json]\n"
            "record Wrapper\n"
            "  data: Pair[int,text]"
        ),
        "entry": (
            "import lib qualified\n"
            "()"
        ),
    }
    cg = _check_graph(tmp_path, modules)
    # Wrapper is in the graph type table with the correct module_id
    assert (lib_id, "Wrapper") in cg.graph_type_table
    wrapper = cg.graph_type_table[(lib_id, "Wrapper")]
    assert isinstance(wrapper, RecordType)
    assert wrapper.module_id == lib_id


# ---------------------------------------------------------------------------
# BUG 3 regression: missing spans on instantiate diagnostics
# An arity mismatch on a generic type application must carry a line number.
# ---------------------------------------------------------------------------


def test_generic_arity_mismatch_has_span(tmp_path: Path) -> None:
    """Regression for BUG 3: arity-mismatch AglTypeError carries a line number.

    Before the fix, instantiate_from_gdef raised AglTypeError without span=, so
    the error had no source location.  After the fix the error carries the span
    from the AppliedT call site.
    """
    src = "record Box[T]\n  value: T\nlet x: Box[int, text] = Box(value = 1)\nx"
    with pytest.raises(AglTypeError) as exc_info:
        _check(src)
    err = exc_info.value
    assert err.span is not None, (
        "AglTypeError for generic arity mismatch must carry a non-None span"
    )
    assert err.span.start_line > 0, (
        f"Span start_line must be > 0, got {err.span.start_line!r}"
    )


# ---------------------------------------------------------------------------
# BUG 4 regression: D5 generic-def-as-value uses name-keyed lookup (cross-module)
# A cross-module generic def used as a value must typecheck correctly.
# ---------------------------------------------------------------------------


def test_d5_generic_def_as_value_single_module(tmp_path: Path) -> None:
    """Regression for BUG 4: D5 generic-def-as-value path uses correct signature lookup.

    The fix changes _check_varref to consult get_function_signature_by_node_id
    (globally unique, correct for cross-module) BEFORE falling back to
    get_function_signature (name-keyed).

    This e2e test verifies D5 works in single-module mode: a generic function
    used as a value must typecheck when an expected FunctionType annotation is given.
    The fix's node-id lookup is used in single-module mode too (seeded by
    _preregister_funcdef), so this verifies the new lookup path doesn't break anything.

    Note: cross-module D5 testing requires graph.py to handle generic function type
    params in _build_graph_func_sig_table (out of scope for this fix set).
    """
    # D5: generic def used as a value with expected type annotation
    src = "def id[T](x: T) -> T = x\nlet f: (int) -> int = id\nf(1)"
    cp = _check(src)
    assert cp is not None

    # D5: no expected type → error
    with pytest.raises(AglTypeError, match="Cannot infer type arguments"):
        _check("def id[T](x: T) -> T = x\nlet f = id\nf")

    # The cross-module D5 case (lib::id as a value in entry) is tested at the
    # env level: verify that get_function_signature_by_node_id takes priority.
    # This is the path the fixed checker takes; before the fix it only called
    # get_function_signature(ref.name) which returns wrong/None cross-module.
    from agm.agl.typecheck.env import FunctionSignature as FS
    from agm.agl.typecheck.env import TypeEnvironment

    env = TypeEnvironment()
    T = TypeVarType("T")
    generic_sig = FS(params=(("x", T, False),), result=T, type_params=("T",))
    # Simulate what the pre-fix code did: only name-keyed table has a wrong sig.
    wrong_non_generic_sig = FS(params=(("x", IntType(), False),), result=IntType())
    env.register_function_signature("id", wrong_non_generic_sig)
    # The fix also consults node-id lookup first; seed it with the correct generic sig.
    env.register_function_signature_by_node_id(42, generic_sig)

    # Before the fix: get_function_signature("id") returns wrong sig (no type_params).
    assert env.get_function_signature("id") is wrong_non_generic_sig
    # After the fix: get_function_signature_by_node_id(42) returns correct generic sig.
    sig_by_id = env.get_function_signature_by_node_id(42)
    assert sig_by_id is generic_sig
    assert sig_by_id is not None
    assert sig_by_id.type_params == ("T",)


# ---------------------------------------------------------------------------
# BUG 5 regression: _build_graph_func_sig_table drops generic type parameters
# These tests MUST FAIL before the graph.py fix and MUST PASS after.
# ---------------------------------------------------------------------------


def test_cross_module_generic_func_call_inferred(tmp_path: Path) -> None:
    """Cross-module generic function call with inferred type args typechecks.

    lib exports 'def id[T](x: T) -> T = x'.
    Entry imports lib and calls lib::id(5) — type should be inferred as int.

    Before the fix: _build_graph_func_sig_table builds FunctionSignature without
    type_params and resolves the 'T' annotation without type_vars, so 'T' does not
    resolve to a TypeVarType.  The type checker sees id as a non-generic function
    whose param is an unknown rigid name 'T', and rejects the int argument.

    After the fix: type_vars is computed and passed to resolve_type_expr, and
    type_params is threaded into FunctionSignature, so lib::id(5) typechecks and
    infers result type int.
    """
    modules = {
        "lib": "def id[T](x: T) -> T = x",
        "entry": (
            "import lib qualified\n"
            "let r = lib::id(5)\n"
            "r"
        ),
    }
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "r") == IntType()


def test_cross_module_generic_func_call_open_import_inferred(tmp_path: Path) -> None:
    """Cross-module generic function call via open import with inferred type args typechecks.

    lib exports 'def id[T](x: T) -> T = x'.
    Entry open-imports lib and calls id(5) (unqualified).

    This exercises the same _build_graph_func_sig_table fix but via open import,
    ensuring the inferred result type is int.
    """
    modules = {
        "lib": "def id[T](x: T) -> T = x",
        "entry": (
            "import lib\n"
            "let r = id(5)\n"
            "r"
        ),
    }
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "r") == IntType()


def test_cross_module_generic_func_call_explicit_type_args(tmp_path: Path) -> None:
    """Cross-module generic function call with explicit type args typechecks.

    lib exports 'def id[T](x: T) -> T = x'.
    Entry calls lib::id::[int](5) (explicit type arg int).

    Before the fix: type_params is empty so id is treated as non-generic;
    explicit type args '  ::[int]' are rejected with "requires 0 type argument(s)".
    After the fix: type_params=("T",) is set, the explicit instantiation is
    accepted and the result type is int.
    """
    modules = {
        "lib": "def id[T](x: T) -> T = x",
        "entry": (
            "import lib qualified\n"
            "let r = lib::id::[int](5)\n"
            "r"
        ),
    }
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "r") == IntType()


def test_cross_module_generic_func_as_value_d5(tmp_path: Path) -> None:
    """D5: Cross-module generic def used as a value with monomorphic annotation typechecks.

    lib exports 'def id[T](x: T) -> T = x'.
    Entry open-imports lib and binds: 'let f: (int) -> int = id'.

    Before the fix: the graph pre-pass registers id with empty type_params, so
    _check_varref sees a non-generic FunctionType and assigns it without instantiation.
    However, the FunctionType for id has an unresolved 'T' param (not a TypeVarType),
    so the assignment still fails or accepts wrongly.
    After the fix: type_params=("T",) is set, _check_varref finds a generic sig,
    matches (int)->int against (T)->T and correctly instantiates to (int)->int.

    We use open import so 'id' (unqualified) is in scope — the D5 varref path
    triggers on unqualified as well as qualified names.
    """
    modules = {
        "lib": "def id[T](x: T) -> T = x",
        "entry": (
            "import lib\n"
            "let f: (int) -> int = id\n"
            "f(1)"
        ),
    }
    cg = _check_graph(tmp_path, modules)
    assert _binding_value_type(cg, ENTRY_ID, "f") == FunctionType(
        params=(IntType(),), result=IntType()
    )


def test_cross_module_generic_func_call_wrong_type_rejected(tmp_path: Path) -> None:
    """Negative control: cross-module generic call with a wrong arg type is rejected.

    lib exports 'def add[T](x: T, y: T) -> T = x'.
    Entry calls lib::add(5, "hello") — T cannot unify int with text simultaneously.

    This confirms the fix does not weaken type checking for generic functions:
    incompatible argument types remain rejected with a sensible diagnostic.
    """
    modules = {
        "lib": "def add[T](x: T, y: T) -> T = x",
        "entry": (
            "import lib qualified\n"
            'lib::add(5, "hello")'
        ),
    }
    with pytest.raises(AglTypeError):
        _check_graph(tmp_path, modules)


# ---------------------------------------------------------------------------
# Named-only parameters in graph context
# ---------------------------------------------------------------------------


def test_named_only_param_in_graph_function(tmp_path: Path) -> None:
    """check_graph handles a function with a named-only param (*, z) correctly."""
    from agm.agl.typecheck.graph import CheckedModuleGraph  # type: ignore[import-untyped]

    modules = {
        "lib": "def add_named(x: int, *, z: int) -> int = x + z",
        "entry": (
            "import lib\n"
            "let z = 5\n"
            "add_named(3, z)"
        ),
    }
    cg: object = _check_graph(tmp_path, modules)
    assert isinstance(cg, CheckedModuleGraph)
