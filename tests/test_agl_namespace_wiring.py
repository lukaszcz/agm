"""Focused scope and typecheck coverage for slash namespace contributions."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.matchcompile.diagnostics import qualified_owner_name
from agm.agl.modules.ids import ModuleId
from agm.agl.modules.loader import ModuleGraph
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import AglScopeError
from agm.agl.semantics.types import EnumOwnerFormKind, RecordType
from agm.agl.syntax import QualifierAnchor, QualifierChain, QualifierSegment
from agm.agl.syntax.spans import UNKNOWN_SOURCE, SourceSpan
from agm.agl.typecheck import AglTypeError
from agm.agl.typecheck.program import check_program
from tests._agl_helpers import strip_decl_ids
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

    with pytest.raises((AglScopeError, AglTypeError)):
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

    with pytest.raises(AglScopeError, match="ambiguous"):
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

    with pytest.raises(AglScopeError, match="ambiguous"):
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

    with pytest.raises(AglScopeError, match="ambiguous"):
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

    with pytest.raises(AglScopeError, match="ambiguous"):
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
                "  record X(value: int)\n"
                "end S\n"
                "\n"
                "let item = S::X(value = 1)\n"
                "case item of\n"
                "  | lib::X(value) => value\n"
            ),
            "lib": "record X(value: int)\n",
        },
    )

    with pytest.raises(AglScopeError, match="ambiguous"):
        resolve_program(graph)


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

    with pytest.raises(AglScopeError, match="ambiguous"):
        resolve_program(graph)


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

    with pytest.raises(AglScopeError, match="ambiguous"):
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

    with pytest.raises(AglTypeError, match="both"):
        check_program(resolve_program(graph), base_caps())


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
                "  record T[A](value: A)\n"
                "end S\n"
                "\n"
                "def identity(value: lib::T[int]) -> lib::T[int] = value\n"
            ),
            "lib": "record T[A](value: A)\n",
        },
    )

    with pytest.raises(AglTypeError, match="both"):
        check_program(resolve_program(graph), base_caps())


def test_qualified_type_use_and_import_routes_deduplicate_same_origin(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import lib\nuse /lib as lib\ndef identity(value: lib::T) -> lib::T = value\n",
            "lib": "record T(value: int)\n",
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
            "lib": "record T[A](value: A)\n",
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
    ("record R(value: text)\n", "def identity(value: R) -> R = value\n"),
    ("record R(value: text)\n", 'let decoded = "{\\"value\\":\\"ok\\"}" as R\n'),
    ("record R[A](value: A)\n", "def identity(value: R[int]) -> R[int] = value\n"),
    ("record R[A](value: A)\n", 'let decoded = "{\\"value\\":1}" as R[int]\n'),
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

    with pytest.raises(AglTypeError, match="[Aa]mbiguous"):
        check_program(resolve_program(graph), base_caps())


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


def test_resolve_named_type_rejects_root_use_and_import_tail_collision(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": ("import lib::*\nuse S::*\n\nscope S\n  record R(value: text)\nend S\n"),
            "lib": "record R(value: int)\n",
        },
    )

    checked = check_program(resolve_program(graph), base_caps())

    with pytest.raises(AglTypeError, match="[Aa]mbiguous"):
        checked.modules[graph.entry_id].type_env.resolve_named_type("R")


def test_ambiguous_caught_exception_is_reported_as_ambiguous(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import m/a\nimport m/b\nuse m/a::*\nuse m/b::*\n"
                "let _ = try\n  ()\ncatch Boom as e =>\n  ()\n"
            ),
            "m/a": "exception Boom extends Exception()\n",
            "m/b": "exception Boom extends Exception()\n",
        },
    )

    with pytest.raises(AglTypeError, match="[Aa]mbiguous") as raised:
        check_program(resolve_program(graph), base_caps())

    assert "m/a::Boom" in str(raised.value)
    assert "m/b::Boom" in str(raised.value)
    assert raised.value.span is not None


def test_ambiguous_exception_base_is_reported_as_ambiguous(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import m/a\nimport m/b\nuse m/a::*\nuse m/b::*\n"
                "exception Local extends Boom()\n()\n"
            ),
            "m/a": "exception Boom extends Exception()\n",
            "m/b": "exception Boom extends Exception()\n",
        },
    )

    with pytest.raises(AglTypeError, match="[Aa]mbiguous") as raised:
        check_program(resolve_program(graph), base_caps())

    # The report names the contested base and both contributing routes, and
    # points at the declaration that named it.
    message = str(raised.value)
    assert "Boom" in message
    assert "m/a" in message
    assert "m/b" in message
    assert raised.value.span is not None
    assert raised.value.span.start_line == 5


def test_resolve_named_type_deduplicates_root_routes_to_same_origin(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import lib::*\nuse lib::*\n",
            "lib": "record R(value: int)\n",
        },
    )

    checked = check_program(resolve_program(graph), base_caps())

    resolved = checked.modules[graph.entry_id].type_env.resolve_named_type("R")
    assert strip_decl_ids(resolved) == RecordType("R", module_id=ModuleId.from_path("lib"))

    # Negative control: two routes to *different* origins are not deduplicated.
    rival = make_graph_from_files(
        tmp_path / "rival",
        {
            "entry": "import lib::*\nimport other::*\n",
            "lib": "record R(value: int)\n",
            "other": "record R(value: int)\n",
        },
        default_stdlib=False,
    )
    rival_checked = check_program(resolve_program(rival), base_caps())
    with pytest.raises(AglTypeError, match="[Aa]mbiguous"):
        rival_checked.modules[rival.entry_id].type_env.resolve_named_type("R")


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
            "entry": ("use Outer::*\nuse R::*\n\nscope Outer\n  record R(value: int)\nend Outer\n"),
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

    with pytest.raises(AglScopeError, match="not nameable"):
        resolve_program(graph)


def test_use_rejects_ambiguous_scopes_exposed_by_earlier_uses(tmp_path: Path) -> None:
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
            ),
        },
    )

    with pytest.raises(AglScopeError, match="local scopes") as raised:
        resolve_program(graph)

    assert "First" in str(raised.value)
    assert "Second" in str(raised.value)


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

    with pytest.raises(AglScopeError, match="not nameable"):
        resolve_program(graph)


def test_use_rejects_ambiguous_imported_scopes_exposed_by_earlier_uses(
    tmp_path: Path,
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import left\n"
                "import right\n"
                "use left::Outer::*\n"
                "use right::Outer::*\n"
                "use Shared::*\n"
            ),
            "left": (
                "scope Outer\n"
                "\n"
                "  scope Shared\n"
                "    def left() -> int = 1\n"
                "  end Shared\n"
                "end Outer\n"
            ),
            "right": (
                "scope Outer\n"
                "\n"
                "  scope Shared\n"
                "    def right() -> int = 2\n"
                "  end Shared\n"
                "end Outer\n"
            ),
        },
    )

    with pytest.raises(AglScopeError, match="imported modules") as raised:
        resolve_program(graph)

    assert "left" in str(raised.value)
    assert "right" in str(raised.value)


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

    with pytest.raises(AglScopeError, match="not nameable"):
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

    with pytest.raises(AglScopeError, match="ambiguous") as exc_info:
        resolve_program(graph)

    diagnostic = str(exc_info.value)
    for repair in ("hiding", "longer suffix", "/-anchored", "as"):
        assert repair in diagnostic


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
                "  record T(value: text)\n"
                "end A\n"
                "\n"
                "def keep(value: /A::T) -> /A::T = value\n"
                "keep(/A::T(value = 1))"
            ),
            "A": "record T(value: int)",
        },
    )

    assert check_program(resolve_program(graph), base_caps()).entry_id == graph.entry_id


def test_unanchored_type_scope_and_module_route_clash_requires_an_anchor(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import A\n"
                "\n"
                "scope A\n"
                "  record T(value: text)\n"
                "  def keep(value: A::T) -> A::T = value\n"
                "end A\n"
                "\n"
                "()"
            ),
            "A": "record T(value: int)",
        },
    )

    with pytest.raises(AglTypeError, match="module route") as exc_info:
        check_program(resolve_program(graph), base_caps())

    for repair in ("hiding", "longer suffix", "/-anchored", "as"):
        assert repair in str(exc_info.value)


def test_imported_type_route_keeps_its_missing_member_error_over_a_local_scope(
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
            "A": "record Present()",
        },
    )

    with pytest.raises(AglTypeError, match="not accessible") as exc_info:
        check_program(resolve_program(graph), base_caps())

    assert "'Missing'" in str(exc_info.value)
    assert "'A::'" in str(exc_info.value)


def test_unanchored_generic_type_scope_and_module_route_clash_requires_an_anchor(
    tmp_path: Path,
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import A\n"
                "\n"
                "scope A\n"
                "  record T[V](value: V)\n"
                "  def keep(value: A::T[int]) -> A::T[int] = value\n"
                "end A\n"
                "\n"
                "()"
            ),
            "A": "record T[V](value: V)",
        },
    )

    with pytest.raises(AglTypeError, match="module route"):
        check_program(resolve_program(graph), base_caps())


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


def test_generic_is_test_type_and_module_constructor_member_collision_is_ambiguous(
    tmp_path: Path,
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import support/Owner\n"
                "enum Owner[T] | On\n"
                "let flag: Owner[int] = ::Owner[int]::On\n"
                "flag is Owner::On"
            ),
            "support/Owner": "def On() -> int = 1",
        },
    )

    resolved = resolve_program(graph)
    with pytest.raises(AglTypeError, match="module route"):
        check_program(resolved, base_caps())


def test_is_test_type_and_module_constructor_member_collision_is_ambiguous(
    tmp_path: Path,
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import support/config\n"
                "enum config | On\n"
                "let flag: config = ::config::On\n"
                "flag is config::On"
            ),
            "support/config": "def On() -> int = 1",
        },
    )

    resolved = resolve_program(graph)
    with pytest.raises(AglTypeError) as exc_info:
        check_program(resolved, base_caps())

    diagnostic = str(exc_info.value)
    assert "module route" in diagnostic
    for repair in ("hiding", "longer suffix", "/-anchored", "as"):
        assert repair in diagnostic


def test_nonconstructible_tailed_import_is_not_a_constructor_owner(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {"entry": "import lib::Alias\nAlias::value", "lib": "type Alias = int"},
    )

    with pytest.raises(AglScopeError):
        resolve_program(graph)


def test_current_module_anchor_does_not_resolve_an_imported_constructor_owner(
    tmp_path: Path,
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {"entry": "import library::*\n::Unknown::On", "library": "enum Unknown | On"},
    )

    with pytest.raises(AglScopeError) as exc_info:
        resolve_program(graph)

    message = exc_info.value.to_diagnostic().message
    assert "Unknown" in message
    assert "not defined" in message


def test_invalid_qualified_pattern_and_is_routes_reach_typecheck(tmp_path: Path) -> None:
    for use in (
        "case flag of | ::Unknown::On => 1 | _ => 2",
        "flag is ::Unknown::On",
        "flag is /Unknown::Flag::On",
    ):
        graph = make_graph_from_files(
            tmp_path,
            {"entry": f"enum Flag | On | Off\nlet flag: Flag = Flag::On\n{use}"},
        )
        with pytest.raises(AglTypeError):
            check_program(resolve_program(graph), base_caps())


def test_is_test_does_not_treat_an_imported_enum_owner_as_its_variant_route(
    tmp_path: Path,
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import support/config\nenum config | On\n"
                "let flag: config = config::On\nflag is config::On"
            ),
            "support/config": "enum config | On",
        },
    )

    assert check_program(resolve_program(graph), base_caps()).entry_id == graph.entry_id


@pytest.mark.parametrize(
    ("source", "modules", "expected"),
    [
        (
            "enum Flag | On | Off\nlet flag: Flag = Flag::On\nflag is unknown::Flag::On",
            {},
            "Unknown module",
        ),
        (
            "import remote/config hiding Flag\nenum Flag | On | Off\n"
            "let flag: Flag = Flag::On\n"
            "let result = case flag of\n"
            "  | config::Flag::On => 1\n"
            "  | _ => 2\n"
            "result",
            {"remote/config": "enum Flag | On | Off"},
            "not accessible",
        ),
        (
            "import one/config\nimport two/config\nenum Local | On\n"
            "let flag: Local = Local::On\nflag is config::Flag::On",
            {"one/config": "enum Flag | On", "two/config": "enum Flag | On"},
            "ambiguous",
        ),
    ],
)
def test_qualified_enum_patterns_and_is_tests_keep_resolution_verdicts(
    tmp_path: Path,
    source: str,
    modules: dict[str, str],
    expected: str,
) -> None:
    graph = make_graph_from_files(tmp_path, {"entry": source, **modules})

    with pytest.raises(AglTypeError) as exc_info:
        check_program(resolve_program(graph), base_caps())

    assert expected in str(exc_info.value)


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


def test_anchored_enum_owner_form_preserves_its_route(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import remote/config\nlet flag = /remote/config::Flag::On\nflag",
            "remote/config": "enum Flag | On",
        },
    )
    checked = check_program(resolve_program(graph), base_caps())
    env = checked.modules[graph.entry_id].type_env
    span = SourceSpan(1, 1, 1, 1, 0, 0, UNKNOWN_SOURCE)
    qualifier = QualifierChain(
        anchor=QualifierAnchor.MODULE,
        segments=(QualifierSegment("remote/config", None, span, 0),),
        member="",
        span=span,
        node_id=0,
    )

    form = env.resolve_enum_owner_form(
        kind=EnumOwnerFormKind.QUALIFIED_IMPORT,
        owner_name="Flag",
        module_qualifier=qualifier,
    )

    assert form is not None
    assert form.qualifier_anchored is True
    rendered = qualified_owner_name("Flag", form.module_qualifier, anchored=form.qualifier_anchored)
    assert rendered == "/remote/config::Flag"


def test_qualified_enum_owner_form_rejects_a_non_type_member(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import remote/config\n0",
            "remote/config": "def Flag() -> int = 1",
        },
    )
    checked = check_program(resolve_program(graph), base_caps())
    span = SourceSpan(1, 1, 1, 1, 0, 0, UNKNOWN_SOURCE)
    qualifier = QualifierChain(
        anchor=None,
        segments=(QualifierSegment("remote/config", None, span, 0),),
        member="",
        span=span,
        node_id=0,
    )

    assert (
        checked.modules[graph.entry_id].type_env.resolve_enum_owner_form(
            EnumOwnerFormKind.QUALIFIED_IMPORT,
            "Flag",
            qualifier,
        )
        is None
    )


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


def test_enum_owner_forms_exclude_ambiguous_suffix_routes(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import one/config\n"
                "import two/config\n"
                "let flag = one/config::Flag::On\n"
                "case flag of | one/config::Flag::On => 1 | _ => 2"
            ),
            "one/config": "enum Flag | On | Off",
            "two/config": "enum Flag | On | Off",
        },
    )

    checked = check_program(resolve_program(graph), base_caps())
    forms = checked.modules[graph.entry_id].type_env.enum_owner_forms()

    assert not any(form.module_qualifier == ("config",) for form in forms)
    assert {
        form.module_qualifier
        for form in forms
        if form.owner_name == "Flag" and form.module_qualifier is not None
    } >= {("one", "config"), ("two", "config")}


def test_enum_owner_forms_do_not_cross_repeated_import_routes(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import alpha as X hiding E\n"
                "import alpha\n"
                "import beta as X\n"
                "let value = alpha::E::One\n"
                "case value of | alpha::E::One => 1 | _ => 0\n"
            ),
            "alpha": "enum E | One | Two\n",
            "beta": "enum E | Other\n",
        },
    )

    checked = check_program(resolve_program(graph), base_caps())
    forms = checked.modules[graph.entry_id].type_env.enum_owner_forms()
    alias_forms = [
        form for form in forms if form.owner_name == "E" and form.module_qualifier == ("X",)
    ]

    assert {form.source_module_id for form in alias_forms} == {ModuleId.from_path("beta")}


def test_anchored_constructor_route_never_falls_back_to_a_local_type(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import remote/config as C\nenum C | On\n/C::On",
            "remote/config": "enum C | External",
        },
    )

    with pytest.raises(AglScopeError, match="qualifier") as raised:
        resolve_program(graph)

    assert "'/C'" in str(raised.value)


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
    with pytest.raises(AglScopeError, match="ambiguous"):
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
