"""Focused scope and typecheck coverage for slash namespace contributions."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import HiddenMemberError
from agm.agl.matchcompile import NonExhaustiveIssue, compile_program_matches, render_witness
from agm.agl.modules.ids import ModuleId
from agm.agl.modules.loader import ModuleGraph
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import (
    AglScopeError,
    AmbiguousConstructorError,
    AmbiguousQualificationError,
    ImportedModuleOrigin,
    UnknownMemberError,
    UnknownQualifierError,
    UseDeclarationOrigin,
)
from agm.agl.typecheck import AglTypeError
from agm.agl.typecheck.program import check_program
from tests.agl.ir_harness import (
    base_caps,
    make_graph_from_files,
    make_inline_graph_from_files,
)


def _make_graph_without_prelude(tmp_path: Path, modules: dict[str, str]) -> ModuleGraph:
    return make_inline_graph_from_files(tmp_path, modules, default_stdlib=False)


def test_scope_resolves_suffix_anchor_and_tail_contributions(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import services/primary/config\n"
                "import services/secondary/config\n"
                "import services/primary/config::primary\n"
                "let first = /services/primary/config::primary()\n"
                "let second = config::secondary()\n"
                "let third = primary()\n"
                "third"
            ),
            "services/primary/config": "def primary() -> int = 1",
            "services/secondary/config": "def secondary() -> int = 2",
        },
    )

    resolved = resolve_program(graph)
    entry = resolved.modules[graph.entry_id]
    primary = ModuleId.from_path("services/primary/config")
    secondary = ModuleId.from_path("services/secondary/config")

    assert entry.import_env.unqualified["primary"] == frozenset({(primary, "primary")})
    resolved_modules = {ref.module_id for ref in entry.resolved.resolution.values()}
    assert primary in resolved_modules
    assert secondary in resolved_modules


@pytest.mark.parametrize(
    "expression",
    [
        "case value of | X::E::A(_ as item) => item | _ => 0",
        "value is X::E::A",
    ],
)
def test_constructor_consumers_resolve_whole_use_aliases(tmp_path: Path, expression: str) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use S as X\n"
                "\n"
                "scope S\n"
                "  enum E\n    | A(value: int)\n    | B\n"
                "end S\n"
                "\n"
                "let value: X::E = X::E::A(1)\n"
                f"{expression}\n"
            ),
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_bare_renamed_constructor_must_belong_to_matched_enum(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use S::{A::X as Renamed}\n"
                "\n"
                "scope S\n"
                "  enum A | X\n"
                "  enum B | Y\n"
                "end S\n"
                "\n"
                "let value: S::B = S::B::Y\n"
                "case value of | Renamed => 1 | _ => 0\n"
            ),
        },
    )

    with pytest.raises(AglTypeError):
        check_program(resolve_program(graph), base_caps())


def test_bare_renamed_fieldful_constructor_must_use_a_constructor_pattern(
    tmp_path: Path,
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use S::{E::A as X}\n"
                "\n"
                "scope S\n"
                "  enum E | A(value: int)\n"
                "end S\n"
                "\n"
                "let value: S::E = X(1)\n"
                "case value of | X => 1 | _ => 0\n"
            ),
        },
    )

    with pytest.raises(AglTypeError):
        check_program(resolve_program(graph), base_caps())


def test_constructor_consumers_honor_use_tail_renames(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use S::{E::A as X}\n"
                "\n"
                "scope S\n"
                "  enum E\n    | A(value: int)\n    | B\n"
                "end S\n"
                "\n"
                "let value: S::E = X(1)\n"
                "let selected = case value of | X(_ as item) => item | _ => 0\n"
                "let matches = value is X\n"
                "selected\n"
            ),
        },
    )

    checked = check_program(resolve_program(graph), base_caps())
    constructors = tuple(checked.modules[graph.entry_id].resolved.constructor_refs.values())
    assert any(constructor.owner_name == "A" for constructor in constructors)


def test_ambiguous_whole_use_alias_constructor_qualifier_is_rejected(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use First as X\n"
                "use Second as X\n"
                "\n"
                "scope First\n"
                "  enum E | A\n"
                "end First\n"
                "\n"
                "scope Second\n"
                "  enum E | A\n"
                "end Second\n"
                "\n"
                "let value = First::E::A\n"
                "value is X::E::A\n"
            ),
        },
    )

    with pytest.raises(AmbiguousQualificationError):
        check_program(resolve_program(graph), base_caps())


@pytest.mark.parametrize(
    "expression",
    [
        "case value of | E::A(_ as item) => item | _ => 0",
        "value is E::A",
    ],
)
def test_constructor_consumers_honor_use_hiding(tmp_path: Path, expression: str) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use S::* hiding E::A\n"
                "\n"
                "scope S\n"
                "  enum E\n    | A(value: int)\n    | B\n"
                "end S\n"
                "\n"
                "let value = S::E::A(1)\n"
                f"{expression}\n"
            ),
        },
    )

    with pytest.raises((AglScopeError, AglTypeError)):
        check_program(resolve_program(graph), base_caps())


def test_use_contributions_keep_type_and_value_namespaces_separate(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use Types::*\n"
                "use Values::*\n"
                "\n"
                "scope Types\n"
                "  enum T\n    | member\n"
                "end Types\n"
                "\n"
                "scope Values\n"
                "  def T() -> int = 7\n"
                "end Values\n"
                "\n"
                "T()\n"
            ),
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_import_tails_keep_type_and_value_namespaces_separate(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import types::*\nimport values::*\nT()\n",
            "types": "enum T\n  | member\n",
            "values": "def T() -> int = 7\n",
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_root_use_value_ignores_same_named_imported_type(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import types::*\n"
                "use Values::*\n"
                "\n"
                "scope Values\n"
                "  record T\n"
                "    value: int\n"
                "end Values\n"
                "\n"
                "T(7)\n"
            ),
            "types": "enum T | member\n",
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_imported_value_ignores_same_named_root_use_type(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import values::*\n"
                "use Types::*\n"
                "\n"
                "scope Types\n"
                "  enum T | member\n"
                "end Types\n"
                "\n"
                "T()\n"
            ),
            "values": "def T() -> int = 7\n",
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_inner_type_only_use_does_not_hide_root_imported_value(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import values::{x}\n"
                "\n"
                "scope Inner\n"
                "  use Types::{x}\n"
                "  def call() -> int = x()\n"
                "end Inner\n"
                "\n"
                "scope Types\n"
                "  enum x | member\n"
                "end Types\n"
                "\n"
                "Inner::call()\n"
            ),
            "values": "def x() -> int = 7\n",
        },
    )

    resolved = resolve_program(graph)

    x_refs = [
        ref
        for ref in resolved.modules[graph.entry_id].resolved.resolution.values()
        if ref.name == "x"
    ]
    assert {ref.module_id for ref in x_refs} == {ModuleId.from_path("values")}
    check_program(resolved, base_caps())


def test_inner_type_only_use_preserves_outer_value_ambiguity(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import left::{x}\n"
                "import right::{x}\n"
                "\n"
                "scope Inner\n"
                "  use Types::{x}\n"
                "  def call() -> int = x()\n"
                "end Inner\n"
                "\n"
                "scope Types\n"
                "  enum x | member\n"
                "end Types\n"
                "\n"
                "Inner::call()\n"
            ),
            "left": "def x() -> int = 1\n",
            "right": "def x() -> int = 2\n",
        },
    )

    with pytest.raises(AmbiguousQualificationError):
        resolve_program(graph)


def test_type_use_lookup_continues_past_inner_value_contribution(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use Types::*\n"
                "\n"
                "scope Inner\n"
                "  use Values::*\n"
                "  def identity(value: T) -> T = value\n"
                "end Inner\n"
                "\n"
                "scope Types\n"
                "  enum T\n    | member\n"
                "end Types\n"
                "\n"
                "scope Values\n"
                "  def T() -> int = 7\n"
                "end Values\n"
                "\n"
                "Inner::identity(Types::T::member)\n"
            ),
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_constructor_lookup_continues_past_inner_value_contribution(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use Types::{E::A as X}\n"
                "\n"
                "scope Inner\n"
                "  use Values::{X}\n"
                "  def selected(value: Types::E) -> int =\n"
                "    case value of | X => 1 | _ => 0\n"
                "  def matches(value: Types::E) -> bool = value is X\n"
                "  def call() -> int = X()\n"
                "end Inner\n"
                "\n"
                "scope Types\n"
                "  enum E | A\n"
                "end Types\n"
                "\n"
                "scope Values\n"
                "  def X() -> int = 7\n"
                "end Values\n"
                "\n"
                "Inner::selected(Types::E::A)\n"
            ),
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_unbraced_single_member_use_tail_can_be_renamed(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use Scope::member as Alias\n"
                "\n"
                "scope Scope\n"
                "  def member() -> int = 1\n"
                "end Scope\n"
                "\n"
                "Alias()\n"
            ),
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_whole_target_alias_preserves_an_empty_local_scope(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": ("use Empty as Alias\nuse Alias::*\n\nscope Empty\nend Empty\n\n()\n"),
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_local_use_can_expose_empty_nested_scope(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use Outer::{Empty}\n"
                "use Empty::*\n"
                "\n"
                "scope Outer\n"
                "\n"
                "  scope Empty\n"
                "  end Empty\n"
                "end Outer\n"
                "\n"
                "()\n"
            ),
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_local_use_can_rename_empty_nested_scope(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use Outer::{Empty as E}\n"
                "use E::*\n"
                "\n"
                "scope Outer\n"
                "\n"
                "  scope Empty\n"
                "  end Empty\n"
                "end Outer\n"
                "\n"
                "()\n"
            ),
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_single_member_alias_can_follow_local_scope_alias(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use Outer as O\n"
                "use O::member as Alias\n"
                "\n"
                "scope Outer\n"
                "  def member() -> int = 1\n"
                "end Outer\n"
                "\n"
                "Alias()\n"
            ),
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_single_member_alias_can_follow_imported_scope_alias(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import lib\nuse lib::Outer as O\nuse O::member as Alias\nAlias()\n",
            "lib": "scope Outer\n  def member() -> int = 1\nend Outer\n",
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_unbraced_ordered_binding_use_tail_can_be_renamed(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use Scope::member as Alias\n\nscope Scope\n  var member = 1\nend Scope\n\nAlias\n"
            ),
        },
    )

    check_program(resolve_program(graph), base_caps())


@pytest.mark.parametrize(
    "use_decl",
    (
        "import lib\nuse /lib::Scope::member as Alias\n",
        "import lib::{Scope::member}\nuse Scope::member as Alias\n",
    ),
)
def test_unbraced_imported_member_use_tail_can_be_renamed(tmp_path: Path, use_decl: str) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": f"{use_decl}Alias()\n",
            "lib": "scope Scope\n  def member() -> int = 1\nend Scope\n",
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_root_local_use_and_import_tail_collision_is_ambiguous(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": ("import lib::{x}\nuse S::*\n\nscope S\n  def x() -> int = 2\nend S\n\nx()\n"),
            "lib": "def x() -> int = 1\n",
        },
    )

    with pytest.raises(AmbiguousQualificationError):
        resolve_program(graph)


def test_qualified_use_and_import_route_collision_is_ambiguous(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import lib\nuse S as lib\n\nscope S\n  def x() -> int = 2\nend S\n\nlib::x()\n"
            ),
            "lib": "def x() -> int = 1\n",
        },
    )

    with pytest.raises(AmbiguousQualificationError):
        resolve_program(graph)


def test_qualified_use_collision_preserves_ambiguous_import_verdict(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import a/lib\n"
                "import b/lib\n"
                "use S as lib\n"
                "\n"
                "scope S\n"
                "  def x() -> int = 3\n"
                "end S\n"
                "\n"
                "lib::x()\n"
            ),
            "a/lib": "def x() -> int = 1\n",
            "b/lib": "def x() -> int = 2\n",
        },
    )

    with pytest.raises(AmbiguousQualificationError):
        resolve_program(graph)


def test_qualified_use_and_import_route_deduplicate_the_same_origin(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import lib\nuse /lib as lib\nlib::x()\n",
            "lib": "def x() -> int = 1\n",
        },
    )

    resolve_program(graph)


def test_qualified_constructor_use_and_import_route_collision_is_ambiguous(
    tmp_path: Path,
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import lib\n"
                "use S as lib\n"
                "\n"
                "scope S\n"
                "  record X\n"
                "    value: int\n"
                "end S\n"
                "\n"
                "let item = S::X(value = 1)\n"
                "case item of\n"
                "  | lib::X(value) => value\n"
            ),
            "lib": "record X\n  value: int\n",
        },
    )

    with pytest.raises(AmbiguousQualificationError):
        resolve_program(graph)


def test_qualified_constructor_pattern_use_and_import_routes_deduplicate_same_origin(
    tmp_path: Path,
) -> None:
    """The same deduplication applies to a pattern spelling."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import lib\n"
                "use /lib as lib\n"
                "\n"
                "let item = lib::X(value = 1)\n"
                "case item of\n"
                "  | lib::X(value) => value\n"
            ),
            "lib": "record X\n  value: int\n",
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_qualified_constructor_is_use_and_import_routes_deduplicate_same_origin(
    tmp_path: Path,
) -> None:
    """The same deduplication applies to an ``is`` spelling."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import lib\nuse /lib as lib\n\nlet item: lib::E = lib::E::A\nitem is lib::E::A\n"
            ),
            "lib": "enum E | A\n",
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_nested_constructor_use_and_import_route_collision_is_ambiguous(
    tmp_path: Path,
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": ("import lib\nuse S as lib\n\nscope S\n  enum E | A\nend S\n\nlib::E::A\n"),
            "lib": "enum E | A\n",
        },
    )

    with pytest.raises(AmbiguousQualificationError):
        resolve_program(graph)


def test_constructor_pattern_route_ambiguous_among_imports_and_a_use_declaration(
    tmp_path: Path,
) -> None:
    """A route two imports alias identically, and a ``use`` declares too, carries one
    :class:`ImportedModuleOrigin` per import plus a :class:`UseDeclarationOrigin`,
    at an ``is`` spelling (never a value spelling, which resolves first)."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import lib1 as lib\n"
                "import lib2 as lib\n"
                "use S as lib\n"
                "\n"
                "scope S\n"
                "  enum E | A\n"
                "end S\n"
                "\n"
                "let x: S::E = S::E::A\n"
                "x is lib::E::A\n"
            ),
            "lib1": "enum E | A\n",
            "lib2": "enum E | A\n",
        },
    )

    with pytest.raises(AmbiguousQualificationError) as excinfo:
        resolve_program(graph)
    entry_id = graph.entry_id
    assert set(excinfo.value.origins) == {
        UseDeclarationOrigin((entry_id, ("S", "E", "A"))),
        ImportedModuleOrigin((ModuleId.from_path("lib1"), ("E", "A"))),
        ImportedModuleOrigin((ModuleId.from_path("lib2"), ("E", "A"))),
    }


def test_qualified_pattern_with_colliding_use_routes_is_ambiguous(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use First as X\n"
                "use Second as X\n"
                "\n"
                "scope First\n"
                "  enum E | A\n"
                "end First\n"
                "\n"
                "scope Second\n"
                "  enum E | A\n"
                "end Second\n"
                "\n"
                "let value = First::E::A\n"
                "case value of | X::E::A => 1 | _ => 0\n"
            ),
        },
    )

    with pytest.raises(AmbiguousQualificationError):
        resolve_program(graph)


def test_qualified_type_use_and_import_route_collision_is_ambiguous(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import lib\n"
                "use S as lib\n"
                "\n"
                "scope S\n"
                "  type T = int\n"
                "end S\n"
                "\n"
                "def identity(value: lib::T) -> lib::T = value\n"
            ),
            "lib": "type T = text\n",
        },
    )

    with pytest.raises(AmbiguousQualificationError):
        resolve_program(graph)


def test_ambiguous_qualification_reports_two_module_origins(tmp_path: Path) -> None:
    """A bare name two imported modules both expose carries one
    :class:`ImportedModuleOrigin` per module, never a use-declaration origin."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import one/config::*\nimport two/config::*\nshared()\n",
            "one/config": "def shared() -> int = 1\n",
            "two/config": "def shared() -> int = 2\n",
        },
    )

    with pytest.raises(AmbiguousQualificationError) as excinfo:
        resolve_program(graph)

    assert set(excinfo.value.origins) == {
        ImportedModuleOrigin((ModuleId.from_path("one/config"), "shared")),
        ImportedModuleOrigin((ModuleId.from_path("two/config"), "shared")),
    }


def test_ambiguous_qualification_reports_two_use_declaration_origins(tmp_path: Path) -> None:
    """A bare name two local ``use`` declarations both expose carries one
    :class:`UseDeclarationOrigin` per declaration, naming each's own scope."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use A::*\n"
                "use B::*\n"
                "\n"
                "scope A\n  def shared() -> int = 1\nend A\n"
                "\n"
                "scope B\n  def shared() -> int = 2\nend B\n"
                "\n"
                "shared()\n"
            ),
        },
    )

    with pytest.raises(AmbiguousQualificationError) as excinfo:
        resolve_program(graph)

    entry_id = graph.entry_id
    assert set(excinfo.value.origins) == {
        UseDeclarationOrigin((entry_id, ("A", "shared"))),
        UseDeclarationOrigin((entry_id, ("B", "shared"))),
    }


def test_ambiguous_qualification_reports_mixed_module_and_use_origins(tmp_path: Path) -> None:
    """A bare name one import and one local ``use`` both expose carries one
    origin of each kind."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import lib::*\nuse S::*\n\nscope S\n  def shared() -> int = 2\nend S\n\nshared()\n"
            ),
            "lib": "def shared() -> int = 1\n",
        },
    )

    with pytest.raises(AmbiguousQualificationError) as excinfo:
        resolve_program(graph)

    entry_id = graph.entry_id
    assert set(excinfo.value.origins) == {
        ImportedModuleOrigin((ModuleId.from_path("lib"), "shared")),
        UseDeclarationOrigin((entry_id, ("S", "shared"))),
    }


def test_ambiguous_qualification_reports_two_region_scoped_import_origins(
    tmp_path: Path,
) -> None:
    """A bare name two region-scoped import tails both expose carries one
    :class:`ImportedModuleOrigin` per module -- recorded provenance, not a
    scope-region-shaped guess, exactly as it is at the root."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "scope S\n"
                "  import one/config::*\n"
                "  import two/config::*\n"
                "  let y = shared()\n"
                "end S\n"
            ),
            "one/config": "def shared() -> int = 1\n",
            "two/config": "def shared() -> int = 2\n",
        },
    )

    with pytest.raises(AmbiguousQualificationError) as excinfo:
        resolve_program(graph)

    assert set(excinfo.value.origins) == {
        ImportedModuleOrigin((ModuleId.from_path("one/config"), "shared")),
        ImportedModuleOrigin((ModuleId.from_path("two/config"), "shared")),
    }


def test_ambiguous_qualification_reports_two_region_use_declaration_origins(
    tmp_path: Path,
) -> None:
    """A bare name two region-scoped ``use`` declarations both expose carries
    one :class:`UseDeclarationOrigin` per declaration, naming each's own
    nested scope path -- recorded provenance, not a guess, inside a region
    exactly as it is at the root."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "scope S\n"
                "  use A::*\n"
                "  use B::*\n"
                "\n"
                "  scope A\n    def shared() -> int = 1\n  end A\n"
                "\n"
                "  scope B\n    def shared() -> int = 2\n  end B\n"
                "\n"
                "  let y = shared()\n"
                "end S\n"
            ),
        },
    )

    with pytest.raises(AmbiguousQualificationError) as excinfo:
        resolve_program(graph)

    entry_id = graph.entry_id
    assert set(excinfo.value.origins) == {
        UseDeclarationOrigin((entry_id, ("S", "A", "shared"))),
        UseDeclarationOrigin((entry_id, ("S", "B", "shared"))),
    }


def test_ambiguous_qualification_reports_mixed_origins_inside_a_region(
    tmp_path: Path,
) -> None:
    """A bare name a region-scoped import tail and a region-scoped ``use``
    both expose carries one origin of each kind, provenance-tagged even
    though both contributions are recorded inside the same nested region."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "scope S\n"
                "  import lib::*\n"
                "  use T::*\n"
                "\n"
                "  scope T\n    def shared() -> int = 2\n  end T\n"
                "\n"
                "  let y = shared()\n"
                "end S\n"
            ),
            "lib": "def shared() -> int = 1\n",
        },
    )

    with pytest.raises(AmbiguousQualificationError) as excinfo:
        resolve_program(graph)

    entry_id = graph.entry_id
    assert set(excinfo.value.origins) == {
        ImportedModuleOrigin((ModuleId.from_path("lib"), "shared")),
        UseDeclarationOrigin((entry_id, ("S", "T", "shared"))),
    }


def test_use_tail_names_a_member_the_target_scope_does_not_declare(
    tmp_path: Path,
) -> None:
    """A ``use S::{Missing}`` tail naming no member of ``S`` raises
    :class:`UnknownMemberError` -- ``S`` itself resolves, so the target is
    known and only its selected member is missing, unlike an unresolved
    qualifier route."""
    graph = make_graph_from_files(
        tmp_path,
        {"entry": "use S::{Missing}\n\nscope S\n  def present() -> int = 1\nend S\n"},
    )

    with pytest.raises(UnknownMemberError):
        resolve_program(graph)


def test_use_opened_enum_route_naming_a_sibling_declaration_is_unknown_member(
    tmp_path: Path,
) -> None:
    """A ``use``-opened enum route, spelled as if constructing a sibling
    declaration, raises :class:`UnknownMemberError` -- the route itself
    resolves through ``use``, so a spelling it does not select is a missing
    constructor, not an unresolvable qualifier."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use S::{Color}\n\n"
                "scope S\n  enum Color\n    | Red\n  record Shape\n    x: int\nend S\n\n"
                "Color::Shape"
            )
        },
    )

    with pytest.raises(UnknownMemberError):
        resolve_program(graph)


def test_bare_module_root_qualifier_naming_no_declaration_is_unknown_member(
    tmp_path: Path,
) -> None:
    """``::nope`` naming nothing at the module root raises
    :class:`UnknownMemberError` -- the module root always resolves, so a name
    it does not declare is a missing member, not an unresolvable qualifier."""
    graph = make_graph_from_files(tmp_path, {"entry": "::nope\n"})

    with pytest.raises(UnknownMemberError):
        resolve_program(graph)


def test_ambiguous_record_constructor_reports_module_qualified_origins(
    tmp_path: Path,
) -> None:
    """A bare record constructor two imported modules both declare carries
    one :class:`ImportedModuleOrigin` per module, exactly as an ambiguous
    enum member does -- :class:`AmbiguousConstructorError` is built through
    the same origin-recording constructor-ambiguity path regardless of the
    owning type's own kind."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import a/lib::*\nimport b/lib::*\nlet probe = Point(x = 1)\n",
            "a/lib": "record Point\n  x: int\n",
            "b/lib": "record Point\n  x: int\n",
        },
    )

    with pytest.raises(AmbiguousConstructorError) as excinfo:
        resolve_program(graph)

    assert set(excinfo.value.origins) == {
        ImportedModuleOrigin((ModuleId.from_path("a/lib"), "Point")),
        ImportedModuleOrigin((ModuleId.from_path("b/lib"), "Point")),
    }
    assert excinfo.value.repair == "a/lib::Point"


@pytest.mark.parametrize(
    "own",
    [
        pytest.param("", id="route"),
        pytest.param("scope lib\n  def Point() -> int = 1\nend lib\n", id="own-scope-of-the-route"),
    ],
)
def test_ambiguous_record_constructor_repair_selects_it_where_written(
    tmp_path: Path, own: str
) -> None:
    """The repair of an ambiguous bare constructor selects the first candidate
    where the bare spelling was written, even when an own scope claims its
    shortest route spelling."""
    modules = {"lib": "record Point\n  x: int\n", "lib2": "record Point\n  x: int\n"}
    header = f"import lib::*\nimport lib2::*\n{own}"
    with pytest.raises(AmbiguousConstructorError) as excinfo:
        resolve_program(
            make_graph_from_files(
                tmp_path / "ambiguous", {**modules, "entry": f"{header}let p = Point(x = 1)\n"}
            )
        )
    repaired = f"{header}let p: /lib::Point = {excinfo.value.repair}(x = 1)\n"
    check_program(
        resolve_program(
            make_graph_from_files(tmp_path / "repaired", {**modules, "entry": repaired})
        ),
        base_caps(),
    )


def test_ambiguous_routed_owner_selects_unknown_member_when_no_candidate_declares_it(
    tmp_path: Path,
) -> None:
    """A value-position constructor route whose owner alone is ambiguous
    still selects none when neither candidate owner declares the requested
    member: the leading segment's ambiguity at its one step is decided
    full-path-first, so the owner's own ambiguity is not the final verdict
    while the requested member could still disambiguate it -- and here it
    cannot, since no candidate declares it either."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import one/types\nimport two/types\ntypes::Color::NoSuch\n",
            "one/types": "enum Color\n  | Red\n  | Green\n",
            "two/types": "enum Color\n  | Red\n  | Blue\n",
        },
    )

    with pytest.raises(UnknownMemberError):
        resolve_program(graph)


def test_ambiguous_routed_owner_selects_its_one_hidden_candidate(tmp_path: Path) -> None:
    """A routed owner ambiguous in isolation still selects the one candidate

    declaring the requested member even when that very import hides it: the
    member's own visibility is decided only once its owner is, so the hiding
    import's own verdict -- not a further ambiguity or a missing member --
    is what the selected candidate then raises.
    """
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import one/types hiding Color::Green\nimport two/types\ntypes::Color::Green\n"
            ),
            "one/types": "enum Color\n  | Green\n  | Red\n",
            "two/types": "enum Color\n  | Red\n  | Blue\n",
        },
    )

    with pytest.raises(HiddenMemberError):
        resolve_program(graph)


def test_ambiguous_routed_owner_reports_hidden_when_every_candidate_hides_it(
    tmp_path: Path,
) -> None:
    """A routed owner ambiguous in isolation, whose requested member every

    candidate both declares and hides, selects nothing: the member is hidden.
    """
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import one/types hiding Color::Red\n"
                "import two/types hiding Color::Red\n"
                "types::Color::Red\n"
            ),
            "one/types": "enum Color\n  | Red\n  | Green\n",
            "two/types": "enum Color\n  | Red\n  | Blue\n",
        },
    )

    with pytest.raises(HiddenMemberError):
        resolve_program(graph)


def test_ambiguous_bare_owner_selects_its_one_hidden_candidate(tmp_path: Path) -> None:
    """A bare ``use``-opened owner ambiguous in isolation still selects the

    one candidate declaring the requested member even when that very ``use``
    hides it, mirroring the routed case: the member's own visibility is
    decided only once its owner is.
    """
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": ("import m\nimport n\nuse m::* hiding Color::Green\nuse n::*\nColor::Green\n"),
            "m": "enum Color\n  | Red\n  | Green\n",
            "n": "enum Color\n  | Red\n  | Blue\n",
        },
    )

    with pytest.raises(HiddenMemberError):
        resolve_program(graph)


def test_ambiguous_bare_owner_reports_hidden_when_every_candidate_hides_it(
    tmp_path: Path,
) -> None:
    """A bare ``use``-opened owner ambiguous in isolation, whose requested

    member every candidate both declares and hides, selects nothing, mirroring
    the routed case.
    """
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import m\nimport n\n"
                "use m::* hiding Color::Red\n"
                "use n::* hiding Color::Red\n"
                "Color::Red\n"
            ),
            "m": "enum Color\n  | Red\n  | Green\n",
            "n": "enum Color\n  | Red\n  | Blue\n",
        },
    )

    with pytest.raises(HiddenMemberError):
        resolve_program(graph)


def test_qualified_applied_type_use_and_import_route_collision_is_ambiguous(
    tmp_path: Path,
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import lib\n"
                "use S as lib\n"
                "\n"
                "scope S\n"
                "  record T[A]\n"
                "    value: A\n"
                "end S\n"
                "\n"
                "def identity(value: lib::T[int]) -> lib::T[int] = value\n"
            ),
            "lib": "record T[A]\n  value: A\n",
        },
    )

    with pytest.raises(AmbiguousQualificationError):
        resolve_program(graph)


def test_qualified_type_use_route_collides_with_an_ambiguous_import_route(
    tmp_path: Path,
) -> None:
    """A use-contributed type's route sharing a suffix-ambiguous import route is ambiguous.

    ``lib`` uniquely resolves to ``S::T`` through the ``use`` alias, but the
    same spelling also matches ``a/lib`` and ``b/lib`` by suffix -- the
    import side's own ambiguity, not just a single competing import, still
    joins the type owner's candidate set."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import a/lib\n"
                "import b/lib\n"
                "use S as lib\n"
                "\n"
                "scope S\n"
                "  record T\n"
                "    value: int\n"
                "end S\n"
                "\n"
                "def identity(value: lib::T) -> lib::T = value\n"
            ),
            "a/lib": "record T\n  value: int\n",
            "b/lib": "record T\n  value: int\n",
        },
    )

    with pytest.raises(AmbiguousQualificationError):
        resolve_program(graph)


def test_qualified_type_use_and_import_routes_deduplicate_same_origin(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import lib\nuse /lib as lib\ndef identity(value: lib::T) -> lib::T = value\n",
            "lib": "record T\n  value: int\n",
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_qualified_applied_type_routes_deduplicate_same_origin(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import lib\n"
                "use /lib as lib\n"
                "def identity(value: lib::T[int]) -> lib::T[int] = value\n"
            ),
            "lib": "record T[A]\n  value: A\n",
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_inner_use_shadows_root_import_and_use_contributions(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import a::{x}\n"
                "use a::*\n"
                "import b\n"
                "\n"
                "scope Inner\n"
                "  use b::*\n"
                "  def selected() -> int = x()\n"
                "end Inner\n"
                "\n"
                "Inner::selected()\n"
            ),
            "a": "def x() -> int = 1\n",
            "b": "def x() -> int = 2\n",
        },
    )

    resolved = resolve_program(graph)

    x_refs = [
        ref
        for ref in resolved.modules[graph.entry_id].resolved.resolution.values()
        if ref.name == "x"
    ]
    assert {ref.module_id for ref in x_refs} == {ModuleId.from_path("b")}
    check_program(resolved, base_caps())


_BARE_TYPE_USES = [
    ("record R\n  value: text\n", "def identity(value: R) -> R = value\n"),
    ("record R\n  value: text\n", 'let decoded = "{\\"value\\":\\"ok\\"}" as R\n'),
    ("record R[A]\n  value: A\n", "def identity(value: R[int]) -> R[int] = value\n"),
    ("record R[A]\n  value: A\n", 'let decoded = "{\\"value\\":1}" as R[int]\n'),
]


@pytest.mark.parametrize(("declaration", "type_use"), _BARE_TYPE_USES)
def test_root_use_and_import_tail_type_collision_is_ambiguous(
    tmp_path: Path, declaration: str, type_use: str
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": ("import lib::*\nuse S::*\nscope S\n" + declaration + "end S\n" + type_use),
            "lib": declaration,
        },
    )

    with pytest.raises(AmbiguousQualificationError):
        resolve_program(graph)


@pytest.mark.parametrize(("declaration", "type_use"), _BARE_TYPE_USES)
def test_root_use_and_import_tail_type_routes_deduplicate_same_origin(
    tmp_path: Path, declaration: str, type_use: str
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import lib::*\nuse lib::*\n" + type_use,
            "lib": declaration,
        },
    )

    check_program(resolve_program(graph), base_caps())


@pytest.mark.parametrize(("declaration", "type_use"), _BARE_TYPE_USES)
def test_regional_use_type_shadows_root_import_tail(
    tmp_path: Path, declaration: str, type_use: str
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import lib::*\n"
                "import selected\n"
                "scope Inner\n"
                "use selected::*\n" + type_use + "end Inner\n"
            ),
            "lib": declaration,
            "selected": declaration,
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_use_can_target_local_scope_exposed_by_an_earlier_use(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use Outer::*\n"
                "use Inner::*\n"
                "\n"
                "scope Outer\n"
                "  def unrelated() -> int = 0\n"
                "\n"
                "  scope Inner\n"
                "    def value() -> int = 1\n"
                "  end Inner\n"
                "end Outer\n"
                "\n"
                "value()\n"
            ),
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_use_can_target_type_scope_exposed_by_an_earlier_use(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use Outer::*\nuse R::*\n\nscope Outer\n  record R\n    value: int\nend Outer\n"
            ),
        },
    )

    resolve_program(graph)


def test_use_rejects_an_ordinary_member_exposed_by_an_earlier_use(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use Outer::*\nuse value::*\n\nscope Outer\n  def value() -> int = 1\nend Outer\n"
            ),
        },
    )

    with pytest.raises(UnknownQualifierError):
        resolve_program(graph)


def test_use_combines_scopes_exposed_by_earlier_uses(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "use First::*\n"
                "use Second::*\n"
                "use Shared::*\n"
                "\n"
                "scope First\n"
                "\n"
                "  scope Shared\n"
                "    def first() -> int = 1\n"
                "  end Shared\n"
                "end First\n"
                "\n"
                "scope Second\n"
                "\n"
                "  scope Shared\n"
                "    def second() -> int = 2\n"
                "  end Shared\n"
                "end Second\n"
                "\n"
                "first() + second()\n"
            ),
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_use_can_target_imported_scope_exposed_by_an_earlier_use(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": ("import library\nuse library::Outer::*\nuse Inner::*\nmember()\n"),
            "library": (
                "scope Outer\n"
                "\n"
                "  scope Inner\n"
                "    def member() -> int = 1\n"
                "  end Inner\n"
                "end Outer\n"
            ),
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_use_can_target_renamed_nested_imported_scope_exposed_by_an_earlier_use(
    tmp_path: Path,
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import library\n"
                "use library::Outer::{Inner::Nested as Selected}\n"
                "use Selected::*\n"
                "member()\n"
            ),
            "library": (
                "scope Outer\n"
                "\n"
                "  scope Inner\n"
                "\n"
                "    scope Nested\n"
                "      def member() -> int = 1\n"
                "    end Nested\n"
                "  end Inner\n"
                "end Outer\n"
            ),
        },
    )

    check_program(resolve_program(graph), base_caps())


def test_use_cannot_target_imported_scope_hidden_by_an_earlier_use(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": ("import library\nuse library::Outer::* hiding Inner\nuse Inner::*\n"),
            "library": (
                "scope Outer\n"
                "\n"
                "  scope Inner\n"
                "    def member() -> int = 1\n"
                "  end Inner\n"
                "end Outer\n"
            ),
        },
    )

    with pytest.raises(HiddenMemberError):
        resolve_program(graph)


def test_use_combines_imported_scopes_exposed_by_earlier_uses(tmp_path: Path) -> None:
    """Both exposed ``Shared`` scopes combine; a path both declare is ambiguous where used."""
    shared = (
        "scope Outer\n"
        "\n"
        "  scope Shared\n"
        "    def {name}() -> int = 1\n"
        "    def common() -> int = 1\n"
        "  end Shared\n"
        "end Outer\n"
    )
    header = "import left\nimport right\nuse left::Outer::*\nuse right::Outer::*\nuse Shared::*\n"
    modules = {"left": shared.format(name="left"), "right": shared.format(name="right")}

    check_program(
        resolve_program(
            make_graph_from_files(
                tmp_path, {"entry": header + "let x = left() + right()\n", **modules}
            )
        ),
        base_caps(),
    )
    with pytest.raises(AmbiguousQualificationError) as excinfo:
        resolve_program(
            make_graph_from_files(tmp_path, {"entry": header + "let x = common()\n", **modules})
        )

    assert set(excinfo.value.origins) == {
        UseDeclarationOrigin((ModuleId.from_path("left"), ("Outer", "Shared", "common"))),
        UseDeclarationOrigin((ModuleId.from_path("right"), ("Outer", "Shared", "common"))),
    }


def test_use_rejects_ordinary_imported_member_exposed_by_an_earlier_use(
    tmp_path: Path,
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": ("import library\nuse library::Outer::*\nuse member::*\n"),
            "library": "scope Outer\n  def member() -> int = 1\nend Outer\n",
        },
    )

    with pytest.raises(UnknownQualifierError):
        resolve_program(graph)


def test_use_imported_nested_scope_selects_its_relative_public_subtree(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import library\n"
                "use library::Scope::* hiding hidden\n"
                "use library::Scope::{Nested::member as chosen}\n"
                "use library::Scope as Selected\n"
                "def selected() -> int = visible() + Nested::member() + chosen()\n"
                "Selected::Nested::member()"
            ),
            "library": (
                "def leaked() -> int = 0\n"
                "\n"
                "scope Scope\n"
                "  def visible() -> int = 1\n"
                "  def hidden() -> int = 2\n"
                "\n"
                "  scope Nested\n"
                "    def member() -> int = 3\n"
                "  end Nested\n"
                "end Scope\n"
            ),
        },
    )

    assert resolve_program(graph).entry_id == graph.entry_id

    for blocked in ("hidden", "leaked"):
        blocked_graph = make_graph_from_files(
            tmp_path,
            {
                "entry": f"import library\nuse library::Scope::* hiding hidden\n{blocked}()",
                "library": (
                    "def leaked() -> int = 0\n\nscope Scope\n  def hidden() -> int = 2\nend Scope"
                ),
            },
        )
        with pytest.raises(AglScopeError):
            resolve_program(blocked_graph)


def test_scope_rejects_an_ambiguous_suffix_at_the_use_site(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import one/config\nimport two/config\nconfig::shared()",
            "one/config": "def shared() -> int = 1",
            "two/config": "def shared() -> int = 2",
        },
    )

    with pytest.raises(AmbiguousQualificationError) as exc_info:
        resolve_program(graph)

    error = exc_info.value
    assert type(error) is AmbiguousQualificationError
    assert error.spelling == "config::shared"
    assert set(error.origins) == {
        ImportedModuleOrigin((ModuleId.from_path("one/config"), "shared")),
        ImportedModuleOrigin((ModuleId.from_path("two/config"), "shared")),
    }


def test_typecheck_routes_qualified_types_patterns_and_is_tests(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import services/flags/config\n"
                "let flag: /services/flags/config::Flag = /services/flags/config::Flag::On\n"
                "let result = case flag of\n"
                "  | /services/flags/config::Flag::On => 1\n"
                "  | /services/flags/config::Flag::Off => 2\n"
                "flag is /services/flags/config::Flag::On"
            ),
            "services/flags/config": "enum Flag | On | Off",
        },
    )

    checked = check_program(resolve_program(graph), base_caps())

    assert ModuleId.from_path("services/flags/config") in checked.modules


def test_type_anchors_select_module_routes_over_same_named_local_scopes(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import A\n"
                "\n"
                "scope A\n"
                "  record T\n"
                "    value: text\n"
                "end A\n"
                "\n"
                "def keep(value: /A::T) -> /A::T = value\n"
                "keep(/A::T(value = 1))"
            ),
            "A": "record T\n  value: int",
        },
    )

    assert check_program(resolve_program(graph), base_caps()).entry_id == graph.entry_id


@pytest.mark.parametrize(("params", "applied"), [("", "A::T"), ("[V]", "A::T[int]")])
def test_own_scope_types_win_over_a_same_named_module_route(
    tmp_path: Path, params: str, applied: str
) -> None:
    field = "int" if params == "" else "V"
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import A\n"
                "\n"
                "scope A\n"
                f"  record T{params}\n"
                f"    own: {field}\n"
                f"  def keep(value: {applied}) -> {applied} = value\n"
                "end A\n"
                "\n"
                "let kept = ::A::keep(A::T(own = 1))\n"
                "kept.own"
            ),
            "A": f"record T{params}\n  imported: {field}",
        },
    )

    assert check_program(resolve_program(graph), base_caps()).entry_id == graph.entry_id


def test_missing_member_under_own_scope_and_module_route_is_an_unknown_member(
    tmp_path: Path,
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import A\n"
                "\n"
                "scope A\n"
                "  def member() -> int = 1\n"
                "end A\n"
                "\n"
                "def use(value: A::Missing) -> int = 1"
            ),
            "A": "record Present",
        },
    )

    with pytest.raises(UnknownMemberError):
        resolve_program(graph)


def test_type_qualifier_beats_route_without_the_requested_member(tmp_path: Path) -> None:
    """A shared route is irrelevant until it contributes the constructor member."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import support/config\n"
                "enum config | On | Off\n"
                "let flag: config = config::On\n"
                "let result = case flag of\n"
                "  | config::On => 1\n"
                "  | config::Off => 2\n"
                "flag is config::On"
            ),
            "support/config": "def unrelated() -> int = 1",
        },
    )

    assert check_program(resolve_program(graph), base_caps()).entry_id == graph.entry_id


@pytest.mark.parametrize(
    ("owner", "declaration", "applied"),
    [("config", "enum config | On", "config"), ("Owner", "enum Owner[T] | On", "Owner[int]")],
)
def test_is_test_member_under_an_own_enum_wins_over_a_route_function(
    tmp_path: Path, owner: str, declaration: str, applied: str
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                f"import support/{owner}\n{declaration}\n"
                f"let flag: {applied} = ::{applied}::On\nflag is {owner}::On"
            ),
            f"support/{owner}": "def On() -> int = 1",
        },
    )

    assert check_program(resolve_program(graph), base_caps()).entry_id == graph.entry_id


def test_nonconstructible_tailed_import_is_not_a_constructor_owner(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {"entry": "import lib::Alias\nAlias::value", "lib": "type Alias = int"},
    )

    with pytest.raises(UnknownMemberError):
        resolve_program(graph)


def test_current_module_anchor_does_not_resolve_an_imported_constructor_owner(
    tmp_path: Path,
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {"entry": "import library::*\n::Unknown::On", "library": "enum Unknown | On"},
    )

    with pytest.raises(UnknownQualifierError) as exc_info:
        resolve_program(graph)

    assert exc_info.value.qualifier == "::Unknown"


@pytest.mark.parametrize("use", ["case u of | ::Unknown::On => 1 | _ => 0", "u is ::Unknown::On"])
def test_current_module_anchor_does_not_qualify_an_imported_enum_owner_in_patterns(
    tmp_path: Path, use: str
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": f"import library::*\nlet u: Unknown = On\n{use}",
            "library": "enum Unknown | On | Off",
        },
    )

    with pytest.raises(AglScopeError):
        resolve_program(graph)


@pytest.mark.parametrize(
    ("use", "error"),
    [
        ("case flag of | ::Unknown::On => 1 | _ => 2", UnknownQualifierError),
        ("flag is ::Unknown::On", UnknownQualifierError),
        ("flag is ::Unknown::Deep::On", UnknownQualifierError),
        ("flag is /Unknown::Flag::On", UnknownQualifierError),
    ],
)
def test_invalid_qualified_pattern_and_is_routes_are_rejected(
    tmp_path: Path, use: str, error: type[AglScopeError]
) -> None:
    """A current-module path naming nothing fails as its value does; a route owner is checked."""
    graph = make_graph_from_files(
        tmp_path,
        {"entry": f"enum Flag | On | Off\nlet flag: Flag = Flag::On\n{use}"},
    )
    with pytest.raises(error):
        resolve_program(graph)


@pytest.mark.parametrize(
    "use",
    ["let other: config = config::On", "flag is config::On", "case flag of | config::On => 1"],
)
def test_own_enum_member_wins_over_a_same_named_route_injecting_its_member(
    tmp_path: Path, use: str
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                f"import support/config\nenum config | On\nlet flag: config = ::config::On\n{use}"
            ),
            "support/config": "enum config | On",
        },
    )

    assert check_program(resolve_program(graph), base_caps()).entry_id == graph.entry_id


@pytest.mark.parametrize(
    ("source", "modules", "error"),
    [
        (
            "enum Flag | On | Off\nlet flag: Flag = Flag::On\nflag is unknown::Flag::On",
            {},
            UnknownQualifierError,
        ),
        (
            "import remote/config hiding Flag\nenum Flag | On | Off\n"
            "let flag: Flag = Flag::On\n"
            "let result = case flag of\n"
            "  | config::Flag::On => 1\n"
            "  | _ => 2\n"
            "result",
            {"remote/config": "enum Flag | On | Off"},
            HiddenMemberError,
        ),
        (
            # A qualifier ambiguous across two imported modules is scope's
            # decision, the same class as the identical value-position
            # ambiguity: AmbiguousQualificationError, never AglTypeError.
            "import one/config\nimport two/config\nenum Local | On\n"
            "let flag: Local = Local::On\nflag is config::Flag::On",
            {"one/config": "enum Flag | On", "two/config": "enum Flag | On"},
            AmbiguousQualificationError,
        ),
    ],
)
def test_qualified_enum_patterns_and_is_tests_keep_resolution_verdicts(
    tmp_path: Path,
    source: str,
    modules: dict[str, str],
    error: type[AglScopeError],
) -> None:
    graph = make_graph_from_files(tmp_path, {"entry": source, **modules})

    with pytest.raises(error):
        resolve_program(graph)


def test_qualified_import_tail_keeps_the_full_type_surface(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import remote/config::read\n"
                "let value: remote/config::Flag = remote/config::Flag::On\n"
                "value"
            ),
            "remote/config": "def read() -> int = 1\nenum Flag | On",
        },
    )

    assert check_program(resolve_program(graph), base_caps()).entry_id == graph.entry_id


@pytest.mark.parametrize(
    "entry",
    [
        ("import remote/config::read\nrecord Wrapper\n  flag: remote/config::Flag\nWrapper"),
        ("import remote/config::read\ndef inspect(flag: remote/config::Flag) -> int = 1\ninspect"),
    ],
)
def test_qualified_import_tail_keeps_the_full_type_surface_during_prepasses(
    tmp_path: Path, entry: str
) -> None:
    """Type-body and signature pre-passes retain the full qualified surface."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": entry,
            "remote/config": "def read() -> int = 1\nenum Flag | On",
        },
    )

    assert check_program(resolve_program(graph), base_caps()).entry_id == graph.entry_id


def test_anchored_qualified_enum_variant_typechecks(tmp_path: Path) -> None:
    """A fully anchored ``/module::Enum::Variant`` selects the member end-to-end."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import remote/config\nlet flag = /remote/config::Flag::On\nflag",
            "remote/config": "enum Flag | On",
        },
    )

    assert check_program(resolve_program(graph), base_caps()).entry_id == graph.entry_id


def test_qualified_pattern_owner_naming_a_non_enum_type_is_rejected(tmp_path: Path) -> None:
    """A pattern qualifier whose owner resolves to a record, not an enum, is a type error."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import remote/config\n"
                "enum Color = Red | Blue\n"
                "let c: Color = Color::Red\n"
                "case c of\n"
                "  | remote/config::Flag::Flag => 1\n"
                "  | _ => 2"
            ),
            "remote/config": "record Flag\n  x: int",
        },
    )

    with pytest.raises(AglTypeError):
        check_program(resolve_program(graph), base_caps())


def test_pattern_and_is_filter_type_module_routes_by_the_referenced_variant(tmp_path: Path) -> None:
    """A module's enum owner does not route its variants directly."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import support/config\n"
                "enum config | On | Off\n"
                "let flag: config = config::On\n"
                "let result = case flag of\n"
                "  | config::On => 1\n"
                "  | config::Off => 2\n"
                "flag is config::On"
            ),
            "support/config": "enum config | External",
        },
    )

    assert check_program(resolve_program(graph), base_caps()).entry_id == graph.entry_id


def _missing_witnesses(graph: ModuleGraph) -> list[str]:
    """The witnesses of *graph*'s non-exhaustive cases, spelled where written."""
    resolved = resolve_program(graph)
    issues = compile_program_matches(check_program(resolved, base_caps())).issues
    return [
        render_witness(issue.witness, resolved.speller(issue.module_id))
        for issue in issues
        if isinstance(issue, NonExhaustiveIssue)
    ]


def test_witness_does_not_spell_an_ambiguous_suffix_route(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import one/config\n"
                "import two/config\n"
                "def f(flag: one/config::Flag) -> int =\n"
                "  case flag of | one/config::Flag::On => 1\n"
            ),
            "one/config": "enum Flag | On | Off",
            "two/config": "enum Flag | On | Off",
        },
    )

    assert _missing_witnesses(graph) == ["one/config::Flag::Off"]


def test_witness_does_not_cross_repeated_import_routes(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import alpha as X hiding E\n"
                "import alpha\n"
                "import beta as X\n"
                "def f(value: alpha::E) -> int =\n"
                "  case value of | alpha::E::One => 1\n"
            ),
            "alpha": "enum E | One | Two\n",
            "beta": "enum E | Other\n",
        },
    )

    assert _missing_witnesses(graph) == ["alpha::E::Two"]


def test_anchored_constructor_route_never_falls_back_to_a_local_type(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import remote/config as C\nenum C | On\n/C::On",
            "remote/config": "enum C | External",
        },
    )

    with pytest.raises(UnknownQualifierError):
        resolve_program(graph)


def test_spec_suffixes_anchor_and_two_line_bare_full_idiom(tmp_path: Path) -> None:
    """Suffixes filter by member, anchors select exactly, and imports union."""
    graph = _make_graph_without_prelude(
        tmp_path,
        {
            "entry": (
                "import std/config\n"
                "import std/list/config\n"
                "import extra/config hiding retries\n"
                "import utils/api\n"
                "import utils/api::bare\n"
                "let a = config::retries()\n"
                "let b = list/config::opt()\n"
                "let c = /std/config::opt()\n"
                "let d = bare()\n"
                "let e = api::full()\n"
                "e"
            ),
            "std/config": "def retries() -> int = 1\ndef opt() -> int = 2",
            "std/list/config": "def opt() -> int = 3",
            "extra/config": "def retries() -> int = 4\ndef opt() -> int = 5",
            "utils/api": "def bare() -> int = 6\ndef full() -> int = 7",
        },
    )

    resolved = resolve_program(graph)
    entry = resolved.modules[graph.entry_id]
    assert entry.import_env.unqualified["bare"] == frozenset(
        {(ModuleId.from_path("utils/api"), "bare")}
    )


def test_hiding_repairs_a_suffix_ambiguity_and_new_import_makes_it_loud(tmp_path: Path) -> None:
    modules = {
        "one/config": "def opt() -> int = 1",
        "two/config": "def opt() -> int = 2",
    }
    repaired = make_graph_from_files(
        tmp_path,
        {"entry": "import one/config\nimport two/config hiding opt\nconfig::opt()", **modules},
    )
    assert resolve_program(repaired).entry_id == repaired.entry_id

    ambiguous = make_graph_from_files(
        tmp_path,
        {"entry": "import one/config\nimport two/config\nconfig::opt()", **modules},
    )
    with pytest.raises(AmbiguousQualificationError):
        resolve_program(ambiguous)


def test_wildcard_alias_is_a_member_filtered_facade(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import facade/* as api\nlet first = api::first()\napi::second()",
            "facade/one": "def first() -> int = 1",
            "facade/two": "def second() -> int = 2",
        },
    )

    assert resolve_program(graph).entry_id == graph.entry_id


def test_wildcard_facade_use_hiding_stays_hidden(tmp_path: Path) -> None:
    modules = {
        "facade/one": "def first() -> int = 1",
        "facade/two": "def second() -> int = 2",
    }
    visible = make_graph_from_files(
        tmp_path,
        {"entry": "import facade/* as api\nuse api::* hiding first\nsecond()", **modules},
    )
    assert resolve_program(visible).entry_id == visible.entry_id

    hidden = make_graph_from_files(
        tmp_path,
        {"entry": "import facade/* as api\nuse api::* hiding first\nfirst()", **modules},
    )
    with pytest.raises(AglScopeError):
        resolve_program(hidden)


def test_wildcard_facade_use_hiding_keeps_a_variant_constructor_hidden(tmp_path: Path) -> None:
    modules = {"facade/one": "enum E\n  | A(value: int)\n  | B\n"}
    visible = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import facade/* as api\n"
                "use api::* hiding E::A\n"
                "let value = E::B\n"
                "case value of | E::B => 1 | _ => 0\n"
            ),
            **modules,
        },
    )
    checked = check_program(resolve_program(visible), base_caps())
    # Hiding one variant leaves the rest of the enum reachable through the
    # facade: the case pattern still matches the facade module's ``E::B``.
    [matched] = checked.modules[visible.entry_id].pattern_constructor_refs.values()
    assert (matched.owner_name, matched.owner_path, matched.owner_module_id) == (
        "B",
        ("E",),
        ModuleId.from_path("facade/one"),
    )

    hidden = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import facade/* as api\nuse api::* hiding E::A\nlet value = E::A(1)\nvalue\n"
            ),
            **modules,
        },
    )
    with pytest.raises((AglScopeError, AglTypeError)):
        check_program(resolve_program(hidden), base_caps())


# ---------------------------------------------------------------------------
# Enum-variant expansion never contributes a bare type (only a constructor and
# pattern candidate), whatever route bare-exposes the owning enum.
# ---------------------------------------------------------------------------

_OTHER_MODULE = {"other": "record Rec\n  x: int\n"}
_LIB_MODULE = {"pk/lib": "import other\n\nenum E = other::Rec | Other\n"}
_VARIANT_MODULES = {**_OTHER_MODULE, **_LIB_MODULE}


@pytest.mark.parametrize(
    ("modules", "entry"),
    [
        pytest.param(
            _VARIANT_MODULES,
            "import pk/* as F\nuse F::*\nlet f = fn(x: Other) => 1\nf\n",
            id="facade-wildcard-use-inline-variant",
        ),
        pytest.param(
            _VARIANT_MODULES,
            "import pk/* as F\nuse F::*\nlet f = fn(x: Rec) => 1\nf\n",
            id="facade-wildcard-use-reused-variant",
        ),
        pytest.param(
            _VARIANT_MODULES,
            "import pk/lib\nuse pk/lib::*\nlet f = fn(x: Other) => 1\nf\n",
            id="plain-import-use",
        ),
        pytest.param(
            _VARIANT_MODULES,
            "import pk/* as F\nuse F::{E}\nlet f = fn(x: Other) => 1\nf\n",
            id="selective-use",
        ),
        pytest.param(
            _VARIANT_MODULES,
            "import pk/lib::*\nlet f = fn(x: Other) => 1\nf\n",
            id="import-star",
        ),
        pytest.param(
            _VARIANT_MODULES,
            "import pk/lib\nlet f = fn(x: pk/lib::Other) => 1\nf\n",
            id="module-qualified",
        ),
        pytest.param(
            _OTHER_MODULE,
            "import other\n\nenum E = other::Rec | Other\n\nlet f = fn(x: Other) => 1\nf\n",
            id="declaring-module-inline-variant",
        ),
        pytest.param(
            _OTHER_MODULE,
            "import other\n\nenum E = other::Rec | Other\n\nlet f = fn(x: Rec) => 1\nf\n",
            id="declaring-module-reused-variant",
        ),
        pytest.param(
            _VARIANT_MODULES,
            (
                "scope s\n"
                "  import pk/lib::*\n"
                "  def check(x: Other) -> int = 1\n"
                "end s\n"
                "\n"
                "let v = 1\n"
                "v\n"
            ),
            id="region-scoped-import",
        ),
    ],
)
def test_enum_variant_expansion_never_contributes_a_bare_type(
    tmp_path: Path, modules: dict[str, str], entry: str
) -> None:
    """A bare-exposed enum's variant name is unknown in type position, not hidden.

    Variant expansion offers ``Other``/``Rec`` as a constructor and pattern
    candidate only: scope rejects the annotation the same way an undeclared
    type (plain ``AglTypeError``), never as a
    :class:`HiddenMemberError`/:class:`ReferencedMemberError` -- those mean a
    route to a real member exists but is currently blocked, which is not the
    case here since no route ever contributes the name as a type.
    """
    with pytest.raises(AglTypeError) as raised:
        resolve_program(make_graph_from_files(tmp_path, {"entry": entry, **modules}))
    assert type(raised.value) is AglTypeError


def test_enum_variant_expansion_never_contributes_a_type_to_an_alias_target(
    tmp_path: Path,
) -> None:
    """A bare-exposed enum's variant name is unusable as a type alias's target.

    An alias's target is resolved once, in scope, so this exercises the same
    ``contributes_a_type`` filter as a direct type annotation, but through the
    alias-resolution route (:meth:`TypeOwnerIndex.owner`) instead of a plain
    type-position lookup: the alias itself is declared and visible, but its
    target selects nothing, so the alias fails where it is declared the same
    way an annotation naming ``Other`` directly would.
    """
    entry = "import pk/* as F\nuse F::*\ntype Alias = Other\nlet f = fn(x: Alias) => 1\nf\n"
    graph = make_graph_from_files(tmp_path, {"entry": entry, **_VARIANT_MODULES})
    with pytest.raises(AglTypeError) as raised:
        resolve_program(graph)
    assert type(raised.value) is AglTypeError


@pytest.mark.parametrize(
    ("modules", "entry"),
    [
        pytest.param(
            _VARIANT_MODULES,
            "import pk/* as F\nuse F::*\nlet f = fn(x: E) => 1\nf(Other)\n",
            id="facade-wildcard-use-inline-variant",
        ),
        pytest.param(
            _VARIANT_MODULES,
            "import pk/* as F\nuse F::*\nlet f = fn(x: E) => 1\nf(Rec(x=1))\n",
            id="facade-wildcard-use-reused-variant",
        ),
        pytest.param(
            _VARIANT_MODULES,
            "import pk/lib\nuse pk/lib::*\nlet f = fn(x: E) => 1\nf(Other)\n",
            id="plain-import-use",
        ),
        pytest.param(
            _VARIANT_MODULES,
            "import pk/lib::*\nlet f = fn(x: E) => 1\nf(Other)\n",
            id="import-star",
        ),
        pytest.param(
            _VARIANT_MODULES,
            "import pk/lib\nlet f = fn(x: pk/lib::E) => 1\nf(pk/lib::Other)\n",
            id="module-qualified",
        ),
        pytest.param(
            _OTHER_MODULE,
            "import other\n\nenum E = other::Rec | Other\n\nlet f = fn(x: E) => 1\nf(Other)\n",
            id="declaring-module-inline-variant",
        ),
        pytest.param(
            _OTHER_MODULE,
            "import other\n\nenum E = other::Rec | Other\n\nlet f = fn(x: E) => 1\nf(Rec(x=1))\n",
            id="declaring-module-reused-variant",
        ),
        pytest.param(
            _VARIANT_MODULES,
            ("import pk/* as F\nuse F::*\nlet v: E = Other\ncase v of | Other => 1 | _ => 0\n"),
            id="pattern-position",
        ),
        pytest.param(
            _VARIANT_MODULES,
            ("scope s\n  import pk/lib::*\n  def make() -> E = Other\nend s\n\nlet v = 1\nv\n"),
            id="region-scoped-import",
        ),
    ],
)
def test_enum_variant_expansion_still_contributes_a_bare_constructor(
    tmp_path: Path, modules: dict[str, str], entry: str
) -> None:
    """The same bare-exposed variant name resolves fine as a value or pattern.

    Contrasts with :func:`test_enum_variant_expansion_never_contributes_a_bare_type`:
    only the type-position route is suppressed. A selective ``use`` of the
    enum alone (``use F::{E}``) never bare-exposes its variants at all -- not
    even as a value -- so it has no counterpart here.
    """
    graph = make_graph_from_files(tmp_path, {"entry": entry, **modules})
    assert check_program(resolve_program(graph), base_caps()).entry_id == graph.entry_id
