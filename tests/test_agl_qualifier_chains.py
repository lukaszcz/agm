"""Behavioral coverage for expression qualifier-chain syntax."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest

from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.modules.loader import LoadedModule, ModuleGraph
from agm.agl.parser import parse_program
from agm.agl.scope import AglScopeError, ModuleResolution
from agm.agl.scope.imports import (
    QualResolutionFound,
    SingleTarget,
    build_import_env,
    resolve_qualified,
)
from agm.agl.scope.program import resolve_program
from agm.agl.syntax import (
    AssignStmt,
    Block,
    Call,
    Case,
    FuncDef,
    IsTest,
    LetDecl,
    NameT,
    NameTarget,
    QualifierAnchor,
    ScopeRegion,
    VarRef,
)
from agm.agl.syntax.nodes import ConstructorPattern, ImportDecl, static_items
from agm.agl.syntax.qualifiers import enclosing_scope_bases
from agm.agl.syntax.spans import UNKNOWN_SOURCE, SourceId, SourceSpan
from agm.agl.syntax.visitor import walk


def _ref(source: str) -> VarRef:
    program = parse_program(source)
    assert isinstance(program.body, Block)
    (expr,) = program.body.items
    assert isinstance(expr, VarRef)
    return expr


_NodeT = TypeVar("_NodeT")


def _find_nodes(program: object, node_type: type[_NodeT]) -> list[_NodeT]:
    """Collect every node of *node_type* in *program*, in tree order."""
    found: list[_NodeT] = []
    walk(program, lambda node: found.append(node) if isinstance(node, node_type) else None)
    return found


def resolve_inline_entry(source: str) -> ModuleResolution:
    """Resolve inline source through the program-level test helper lazily."""
    from tests.agl.module_graph import resolve_inline_entry as resolve

    return resolve(source)


def _entry_resolution(tmp_path: Path, modules: dict[str, str]) -> ModuleResolution:
    """Resolve a multi-module program and return the entry module's resolution."""
    from tests.agl.ir_harness import make_graph_from_files

    resolved = resolve_program(make_graph_from_files(tmp_path, modules))
    return resolved.modules[resolved.entry_id].resolved


def _resolve_without_loader(modules: dict[str, str]) -> ModuleResolution:
    """Resolve a parsed graph without exercising module loading behavior.

    Every module is inline (``path is None``), so the graph carries no
    dependency edges: the loader's own resolution of import declarations into
    edges is exactly what these tests stand apart from.
    """
    loaded: dict[ModuleId, LoadedModule] = {}
    for path, source in modules.items():
        module_id = ENTRY_ID if path == "entry" else ModuleId.from_path(path)
        program = parse_program(source)
        loaded[module_id] = LoadedModule(
            module_id=module_id,
            program=program,
            path=None,
            source=SourceId(0),
            imports=tuple(
                item for item in static_items(program.body.items) if isinstance(item, ImportDecl)
            ),
            export_decls=(),
            source_text=source,
            spaced_qualifiers=(),
            companion_path=None,
        )
    graph = ModuleGraph(
        modules=loaded,
        entry_id=ENTRY_ID,
        sccs=tuple((module_id,) for module_id in loaded),
        adjacency={module_id: () for module_id in loaded},
    )
    return resolve_program(graph).modules[ENTRY_ID].resolved


def test_generic_enum_constructor_resolves_through_a_reexport_facade(tmp_path: Path) -> None:
    resolved = _entry_resolution(
        tmp_path,
        {
            "entry": "import facade\nfacade::Option[int]::some(value = 1)",
            "facade": "export lib::{Option}",
            "lib": "enum Option[T]\n  | some(value: T)",
        },
    )

    (call,) = _find_nodes(resolved.program, Call)
    constructor = resolved.constructor_refs[call.callee.node_id]
    assert constructor.owner_module_id == ModuleId.from_path("lib")
    assert (constructor.owner_path, constructor.owner_name) == (("Option",), "some")


def test_qualified_expression_keeps_segment_spans_and_type_arguments() -> None:
    ref = _ref("module::Type[int]::member")

    assert ref.qualifier is not None
    assert ref.qualifier.anchor is None
    assert ref.qualifier.member == "member"
    assert [segment.name for segment in ref.qualifier.segments] == ["module", "Type"]
    assert ref.qualifier.segments[0].type_args is None
    assert ref.qualifier.segments[1].type_args is not None
    assert ref.qualifier.segments[0].span.start_offset == 0
    assert ref.qualifier.segments[0].span.end_offset == 6
    assert ref.qualifier.segments[1].span.start_offset == 8
    assert ref.qualifier.segments[1].span.end_offset == 17


def test_qualified_expression_retains_module_and_current_module_anchors() -> None:
    module_ref = _ref("/library::value")
    current_module_ref = _ref("::Type::member")

    assert module_ref.qualifier is not None
    assert module_ref.qualifier.anchor is QualifierAnchor.MODULE
    assert module_ref.qualifier.segments[0].span.start_offset == 1
    assert current_module_ref.qualifier is not None
    assert current_module_ref.qualifier.anchor is QualifierAnchor.CURRENT_MODULE
    assert current_module_ref.qualifier.segments[0].span.start_offset == 2


def test_current_module_anchored_chain_keeps_all_qualifier_segments() -> None:
    ref = _ref("::A::B::C")

    assert ref.qualifier is not None
    assert ref.qualifier.anchor is QualifierAnchor.CURRENT_MODULE
    assert [segment.name for segment in ref.qualifier.segments] == ["A", "B"]
    assert ref.qualifier.member == "C"


def test_qualified_patterns_keep_segment_spans_and_type_arguments() -> None:
    source = "case value of | module::Option[int]::None => 1"
    program = parse_program(source)

    assert isinstance(program.body, Block)
    (case,) = program.body.items
    assert isinstance(case, Case)
    qualifier = case.branches[0].pattern.qualifier
    assert qualifier is not None
    assert qualifier.member == "None"
    assert [segment.name for segment in qualifier.segments] == ["module", "Option"]
    assert qualifier.segments[1].type_args is not None
    assert (
        source[qualifier.segments[0].span.start_offset : qualifier.segments[0].span.end_offset]
        == "module"
    )
    assert (
        source[qualifier.segments[1].span.start_offset : qualifier.segments[1].span.end_offset]
        == "Option[int]"
    )


def test_qualified_type_references_keep_segment_spans_and_type_arguments() -> None:
    source = "let value: module::Option[int]::Result = null"
    program = parse_program(source)

    assert isinstance(program.body, Block)
    (decl,) = program.body.items
    assert isinstance(decl, LetDecl)
    assert isinstance(decl.type_ann, NameT)
    qualifier = decl.type_ann.qualifier
    assert qualifier is not None
    assert qualifier.member == "Result"
    assert [segment.name for segment in qualifier.segments] == ["module", "Option"]
    assert qualifier.segments[1].type_args is not None
    assert (
        source[qualifier.segments[0].span.start_offset : qualifier.segments[0].span.end_offset]
        == "module"
    )
    assert (
        source[qualifier.segments[1].span.start_offset : qualifier.segments[1].span.end_offset]
        == "Option[int]"
    )


def test_qualified_is_tests_keep_segment_spans_and_type_arguments() -> None:
    source = "value is module::Option[int]::None"
    program = parse_program(source)

    assert isinstance(program.body, Block)
    (test,) = program.body.items
    assert isinstance(test, IsTest)
    qualifier = test.qualifier
    assert qualifier is not None
    assert qualifier.member == "None"
    assert [segment.name for segment in qualifier.segments] == ["module", "Option"]
    assert qualifier.segments[1].type_args is not None
    assert (
        source[qualifier.segments[0].span.start_offset : qualifier.segments[0].span.end_offset]
        == "module"
    )
    assert (
        source[qualifier.segments[1].span.start_offset : qualifier.segments[1].span.end_offset]
        == "Option[int]"
    )


@pytest.mark.parametrize(
    "source",
    (
        "let value: First::Second::Third::member = null",
        "let value: Option[int]::Result = null",
        "let value = 1\ncase value of | First::Second::Third::member => 1",
        "let value = 1\nvalue is First::Second::Third::member",
    ),
)
def test_long_qualifier_chains_are_not_rejected_by_legacy_shape_validation(source: str) -> None:
    resolve_inline_entry(source)


@pytest.mark.parametrize(
    "source",
    (
        "::/foo::Baz",
        "::foo/bar::Baz",
        "foo::/bar::Baz",
        "foo::bar/baz::Baz",
    ),
)
def test_nonleading_anchor_and_route_segments_report_route_errors(source: str) -> None:
    with pytest.raises(AglScopeError, match="Only the leading qualifier"):
        resolve_inline_entry(source)


def test_current_module_qualified_assignment_remains_an_assignment_target() -> None:
    program = parse_program("::value := 1")

    assert isinstance(program.body, Block)
    (assignment,) = program.body.items
    assert isinstance(assignment, AssignStmt)
    assert isinstance(assignment.target, NameTarget)
    assert assignment.target.qualifier is not None
    assert assignment.target.qualifier.segments == ()


def test_non_expression_chain_routes_reject_only_invalid_nonleading_routes() -> None:
    with pytest.raises(AglScopeError, match="Only the leading qualifier"):
        resolve_inline_entry("let value = 1\ncase value of | module::owner/name::member => 1")


def test_long_qualified_expression_retains_all_segments() -> None:
    ref = _ref("First::Second::Third::member")

    assert ref.qualifier is not None
    assert [segment.name for segment in ref.qualifier.segments] == ["First", "Second", "Third"]


def test_current_module_anchor_uses_root_binding_without_import_environment() -> None:
    resolution = resolve_inline_entry(
        "def root() -> int = 1\n\nscope Nested\n  def use(root: int) -> int = ::root()\nend Nested"
    )
    program = resolution.program
    nested = next(item for item in program.body.items if isinstance(item, ScopeRegion))
    nested_use = next(item for item in nested.items if isinstance(item, FuncDef))
    assert nested_use.name == "use"
    assert isinstance(nested_use.body, Call)
    assert isinstance(nested_use.body.callee, VarRef)
    assert resolution.resolution[nested_use.body.callee.node_id].kind.name == "function_binding"


def test_generic_constructor_ignores_unrelated_same_named_scope() -> None:
    resolve_inline_entry(
        "scope Other\n"
        "\n"
        "  scope A\n"
        "    def ignored() -> int = 0\n"
        "  end A\n"
        "end Other\n"
        "\n"
        "enum A[T]\n"
        "  | make(value: T)\n"
        "A[int]::make(value = 1)"
    )


def test_current_module_anchored_type_constructor_uses_the_chain_constructor_ref() -> None:
    resolution = resolve_inline_entry("enum Option\n  | some\n::Option::some")
    program = resolution.program
    assert isinstance(program.body, Block)
    expr = program.body.items[-1]
    assert isinstance(expr, VarRef)

    constructor = resolution.constructor_refs[expr.node_id]
    assert (constructor.owner_path, constructor.owner_name) == (("Option",), "some")


@pytest.mark.parametrize("source", ("::Unknown::On", "::Unknown[int]::On"))
def test_current_module_unknown_constructor_owner_reports_its_segment(source: str) -> None:
    program = parse_program(source)
    assert isinstance(program.body, Block)
    expr = program.body.items[-1]
    assert isinstance(expr, VarRef)
    assert expr.qualifier is not None

    with pytest.raises(AglScopeError) as exc_info:
        resolve_inline_entry(source)

    assert "Unknown" in exc_info.value.to_diagnostic().message
    assert exc_info.value.span == expr.qualifier.segments[0].span


def test_scoped_enum_members_and_nested_type_members_run_through_the_full_pipeline() -> None:
    from tests.agl.ir_harness import evaluate_ir_output

    output = evaluate_ir_output(
        "enum Option[T] | none | some(value: T)\n"
        "\n"
        "scope Option\n"
        "  def is-empty(value: Option[int]) -> bool = value is Option[int]::none\n"
        "end Option\n"
        "\n"
        "scope A\n"
        "  enum T[U] | value\n"
        "end A\n"
        "\n"
        "let option: Option[int] = Option[int]::some(value = 1)\n"
        "case option of\n"
        "  | Option[int]::some(value) => print value\n"
        "  | Option[int]::none => print 0\n"
        "print(Option::is-empty(Option[int]::none))\n"
        "let nested: A::T[int] = A::T[int]::value\n"
        "print(nested is A::T[int]::value)"
    )

    assert output == "1\ntrue\ntrue\n"


def test_current_module_anchored_multi_segment_chain_reports_its_unknown_path() -> None:
    with pytest.raises(AglScopeError, match="scope path"):
        resolve_inline_entry("::A::B::C")


def test_bare_self_qualified_undefined_name_hints_a_dollar_suffixed_spelling() -> None:
    """``::exec$`` (a zero-segment self-reference) hints the same fix."""
    with pytest.raises(AglScopeError) as exc_info:
        resolve_inline_entry("::exec$")

    assert "exec $" in exc_info.value.to_diagnostic().message


def test_current_module_unknown_constructor_owner_hints_a_dollar_suffixed_segment() -> None:
    """``::exec$::bar`` names the '$'-suffixed segment; the message hints the fix."""
    with pytest.raises(AglScopeError) as exc_info:
        resolve_inline_entry("::exec$::bar")

    assert "exec $" in exc_info.value.to_diagnostic().message


def test_module_anchored_constructor_chain_never_falls_back_to_a_local_type() -> None:
    with pytest.raises(AglScopeError, match="No module"):
        resolve_inline_entry("enum A | value\n/A::value")


def test_nonconstructible_scoped_type_falls_back_to_the_legacy_constructor_diagnostic() -> None:
    with pytest.raises(AglScopeError):
        resolve_inline_entry("type A::Count = int\nA::Count")


@pytest.mark.parametrize(
    "source",
    (
        "scope Tools\n  def f() -> int = 0\nend Tools\n\nTools[int]::f()",
        "scope Tools\n  def f() -> int = 0\nend Tools\n\nlet value: Tools[int]::T = null",
        (
            "scope Tools\n  def f() -> int = 0\nend Tools\n\nlet value = 1\n"
            "case value of | Tools[int]::f => 1"
        ),
        "scope Tools\n  def f() -> int = 0\nend Tools\n\nvalue is Tools[int]::f",
    ),
)
def test_type_arguments_on_a_plain_scope_are_rejected_in_every_chain_position(
    source: str,
) -> None:
    with pytest.raises(AglScopeError, match="Type arguments cannot be applied to scope"):
        resolve_inline_entry(source)


@pytest.mark.parametrize(
    ("source", "match"),
    (
        ("First::Second::Third::member", "not defined|No module"),
        # Under a real import environment, a first segment carrying type
        # arguments is classified as an attempted module route before any
        # name lookup is attempted, so this reports the invalid route shape
        # rather than an unresolved name.
        ("Type[int]::Second::member", "Type arguments cannot be applied to module route"),
    ),
)
def test_long_expression_qualifier_chain_reports_the_unresolved_name(
    source: str, match: str
) -> None:
    with pytest.raises(AglScopeError, match=match):
        resolve_inline_entry(source)


def test_imported_scoped_enum_owner_retains_its_scope_path_for_is_and_case(
    tmp_path: Path,
) -> None:
    """A pattern-qualified ``lib::A::Status::Good`` keeps ``lib``'s scoped owner."""
    resolution = _entry_resolution(
        tmp_path,
        {
            "entry": (
                "import lib\n"
                "let s: lib::A::Status = lib::A::Status::Good\n"
                "print(s is lib::A::Status::Good)\n"
                "print(case s of\n"
                '  | lib::A::Status::Good => "good"\n'
                '  | lib::A::Status::Bad => "bad")\n'
            ),
            "lib": "scope A\n  enum Status\n    | Good\n    | Bad\nend A\n",
        },
    )

    (is_test,) = _find_nodes(resolution.program, IsTest)
    is_cref = resolution.constructor_refs[is_test.node_id]
    assert (is_cref.owner_path, is_cref.owner_name) == (("A", "Status"), "Good")

    (case_node,) = _find_nodes(resolution.program, Case)
    good_pattern = case_node.branches[0].pattern
    assert isinstance(good_pattern, ConstructorPattern)
    case_cref = resolution.constructor_refs[good_pattern.node_id]
    assert (case_cref.owner_path, case_cref.owner_name) == (("A", "Status"), "Good")


def test_explicit_module_route_use_is_not_masked_by_a_same_named_local_scope() -> None:
    """``use /geo/shapes::Point::*`` reaches the routed scope over a same-named local one."""
    resolution = _resolve_without_loader(
        {
            "entry": (
                "import geo/shapes\n"
                "use /geo/shapes::Point::*\n"
                "\n"
                "scope Point\n"
                "  def area() -> int = 1\n"
                "end Point\n"
                "\n"
                "def foreign() -> text = describe()\n"
                "def local() -> int = Point::area()\n"
            ),
            "geo/shapes": 'scope Point\n  def describe() -> text = "point"\nend Point\n',
        }
    )

    var_refs = _find_nodes(resolution.program, VarRef)
    (describe_ref,) = [node for node in var_refs if node.name == "describe"]
    assert resolution.resolution[describe_ref.node_id].module_id != ENTRY_ID

    (area_ref,) = [node for node in var_refs if node.name == "area"]
    assert resolution.resolution[area_ref.node_id].module_id == ENTRY_ID


def test_use_resolves_aliases_suffixes_anchored_routes_and_nested_scopes(tmp_path: Path) -> None:
    del tmp_path
    _resolve_without_loader(
        {
            "entry": (
                "import pkg/tools\n"
                "import pkg/tools as Alias\n"
                "use tools::{ping as probe, Nested}\n"
                "use Alias::Nested::*\n"
                "use /pkg/tools as P\n"
                "def selected() -> int = probe()\n"
                "def retained-path() -> int = Nested::child()\n"
                "def nested() -> int = child()\n"
                "def renamed-route() -> int = P::ping()\n"
            ),
            "pkg/tools": (
                "def ping() -> int = 1\n\nscope Nested\n  def child() -> int = 2\nend Nested\n"
            ),
        },
    )


def test_use_wildcard_alias_facade_combines_module_members(tmp_path: Path) -> None:
    _entry_resolution(
        tmp_path,
        {
            "entry": "import pkg/* as Facade\nuse Facade::*\nfirst() + second()\n",
            "pkg/a": "def first() -> int = 1\n",
            "pkg/b": "def second() -> int = 2\n",
        },
    )


def test_use_wildcard_alias_facade_preserves_member_ambiguity(tmp_path: Path) -> None:
    modules = {
        "entry": "import pkg/* as Facade\nuse Facade::*\ncommon()\n",
        "pkg/a": "def common() -> int = 1\n",
        "pkg/b": "def common() -> int = 2\n",
    }

    with pytest.raises(AglScopeError, match="ambiguous"):
        _entry_resolution(tmp_path, modules)


def test_use_target_local_module_ambiguity_requires_an_anchor(tmp_path: Path) -> None:
    modules = {
        "Point": "def remote() -> int = 1\n",
        "entry": (
            "import Point\nuse Point::*\n\nscope Point\n  def local() -> int = 2\nend Point\n"
        ),
    }

    del tmp_path
    with pytest.raises(AglScopeError, match="ambiguous") as raised:
        _resolve_without_loader(modules)

    diagnostic = str(raised.value)
    assert "Point" in diagnostic
    assert "/Point" in diagnostic
    assert "::Point" in diagnostic

    _resolve_without_loader(
        {**modules, "entry": modules["entry"].replace("use Point::*", "use /Point::*")}
    )
    _resolve_without_loader(
        {**modules, "entry": modules["entry"].replace("use Point::*", "use ::Point::*")}
    )


@pytest.mark.parametrize(
    "use_decl",
    ("use lib as L", "use lib::* hiding Flag::Ready"),
)
def test_use_aliases_and_hiding_do_not_leak_enum_variants(tmp_path: Path, use_decl: str) -> None:
    with pytest.raises(AglScopeError):
        _entry_resolution(
            tmp_path,
            {
                "entry": f"import lib\n{use_decl}\ndef selected() -> Flag = Ready",
                "lib": "enum Flag\n  | Ready\n  | Waiting",
            },
        )


def test_import_hiding_does_not_reintroduce_bare_enum_variant(tmp_path: Path) -> None:
    with pytest.raises(AglScopeError):
        _entry_resolution(
            tmp_path,
            {
                "entry": "import lib::* hiding Flag::Ready\ndef selected() -> Flag = Ready",
                "lib": "enum Flag\n  | Ready\n  | Waiting",
            },
        )


def test_selective_import_does_not_expose_unselected_nested_scope_to_later_use(
    tmp_path: Path,
) -> None:
    with pytest.raises(AglScopeError, match="not nameable"):
        _entry_resolution(
            tmp_path,
            {
                "entry": "import lib::{A::ok}\nuse A::*\nuse Secret::*",
                "lib": (
                    "scope A\n"
                    "  def ok() -> int = 1\n"
                    "\n"
                    "  scope Secret\n"
                    "    def hidden() -> int = 2\n"
                    "  end Secret\n"
                    "end A"
                ),
            },
        )


def test_selective_import_keeps_unselected_nested_scope_qualified_use_route(
    tmp_path: Path,
) -> None:
    _entry_resolution(
        tmp_path,
        {
            "entry": ("import lib::{A::ok}\nuse A::*\nuse lib::A::Secret::*\nhidden()"),
            "lib": (
                "scope A\n"
                "  def ok() -> int = 1\n"
                "\n"
                "  scope Secret\n"
                "    def hidden() -> int = 2\n"
                "  end Secret\n"
                "end A"
            ),
        },
    )


@pytest.mark.parametrize(
    "entry",
    (
        "import lib\nuse lib::f::*",
        "import lib::{f}\nuse f::*",
    ),
)
def test_use_rejects_ordinary_function_targets(tmp_path: Path, entry: str) -> None:
    with pytest.raises(AglScopeError):
        _entry_resolution(
            tmp_path,
            {
                "entry": entry,
                "lib": "def f() -> int = 1",
            },
        )


def test_use_accepts_one_facade_scope_with_multiple_defining_modules(tmp_path: Path) -> None:
    _entry_resolution(
        tmp_path,
        {
            "entry": (
                "import facade::{Scope}\nuse Scope::*\ndef selected() -> int = alpha() + beta()"
            ),
            "facade": "export source/a::{Scope}\nexport source/b::{Scope}",
            "source/a": "scope Scope\n  def alpha() -> int = 1\nend Scope",
            "source/b": "scope Scope\n  def beta() -> int = 2\nend Scope",
        },
    )


def test_use_keeps_distinct_targets_from_one_module_ambiguous() -> None:
    with pytest.raises(AglScopeError, match="ambiguous"):
        _resolve_without_loader(
            {
                "entry": (
                    "import lib as Scope\n"
                    "import lib::{Scope}\n"
                    "use Scope::*\n"
                    "def selected() -> int = member()"
                ),
                "lib": (
                    "def member() -> int = 1\n\nscope Scope\n  def member() -> int = 2\nend Scope"
                ),
            }
        )


def test_use_target_suffix_ambiguity_has_no_preferred_module_route() -> None:
    modules = {
        "entry": "import one/Target\nimport two/Target\nuse Target::*\n",
        "one/Target": "def first() -> int = 1\n",
        "two/Target": "def second() -> int = 2\n",
    }

    with pytest.raises(AglScopeError, match="ambiguous"):
        _resolve_without_loader(modules)

    _resolve_without_loader(
        {**modules, "entry": modules["entry"].replace("use Target::*", "use /one/Target::*")}
    )


def test_use_bare_contributions_narrow_to_their_scope_region(tmp_path: Path) -> None:
    modules = {
        "lib": "def value() -> int = 1\n",
        "entry": (
            "import lib\n\nscope A\n  use lib::*\n  def available() -> int = value()\nend A\n"
        ),
    }
    del tmp_path
    _resolve_without_loader(modules)

    with pytest.raises(AglScopeError):
        _resolve_without_loader(
            {**modules, "entry": modules["entry"] + "def unavailable() -> int = value()\n"}
        )


def test_use_selection_hiding_and_renames_are_additive() -> None:
    _resolve_without_loader(
        {
            "entry": (
                "use Local::* hiding hidden\n"
                "use Local::{Nested::member as renamed}\n"
                "use Local::shown\n"
                "use Local as L\n"
                "\n"
                "scope Local\n"
                "  def shown() -> int = 1\n"
                "  def hidden() -> int = 2\n"
                "\n"
                "  scope Nested\n"
                "    def member() -> int = 3\n"
                "  end Nested\n"
                "end Local\n"
                "\n"
                "def all-members() -> int = Nested::member()\n"
                "def selected-member() -> int = renamed()\n"
                "def single-member() -> int = shown()\n"
                "def aliased-scope() -> int = L::shown()\n"
            )
        }
    )

    with pytest.raises(AglScopeError, match="not declared"):
        _resolve_without_loader(
            {
                "entry": (
                    "use Local::* hiding missing\n"
                    "\n"
                    "scope Local\n"
                    "  def shown() -> int = 1\n"
                    "end Local\n"
                )
            }
        )


def test_local_use_rename_collision_is_ambiguous_when_used() -> None:
    with pytest.raises(AglScopeError, match="ambiguous"):
        _resolve_without_loader(
            {
                "entry": (
                    "use S::{first as chosen, second as chosen}\n"
                    "\n"
                    "scope S\n"
                    "  def first() -> int = 1\n"
                    "  def second() -> int = 2\n"
                    "end S\n"
                    "\n"
                    "def selected() -> int = chosen()"
                )
            }
        )


def test_region_tailed_import_keeps_routes_global_and_bare_names_regional() -> None:
    modules = {
        "lib": "def value() -> int = 1\n",
        "entry": (
            "scope A\n"
            "  import lib::*\n"
            "  def bare() -> int = value()\n"
            "end A\n"
            "\n"
            "scope B\n"
            "  def routed() -> int = lib::value()\n"
            "end B\n"
        ),
    }
    _resolve_without_loader(modules)

    with pytest.raises(AglScopeError):
        _resolve_without_loader(
            {
                **modules,
                "entry": modules["entry"].replace(
                    "def routed() -> int = lib::value()", "def routed() -> int = value()"
                ),
            }
        )


@pytest.mark.parametrize(
    "modules",
    (
        {
            "entry": "import empty\nuse empty::*",
            "empty": "import dependency",
            "dependency": "def value() -> int = 1",
        },
        {
            "entry": "import lib\nuse lib::Empty::*",
            "lib": "record Empty\n  value: int",
        },
        {
            "entry": "import lib\nuse /lib::Empty::*",
            "lib": "scope Empty\nend Empty",
        },
        {
            "entry": "import lib::{Empty}\nuse Empty::*",
            "lib": "record Empty\n  value: int",
        },
        {
            "entry": "import lib hiding only\nuse lib::*",
            "lib": "def only() -> int = 1",
        },
    ),
)
def test_use_accepts_nameable_targets_with_no_visible_members(
    tmp_path: Path, modules: dict[str, str]
) -> None:
    _entry_resolution(tmp_path, modules)


def test_unnameable_use_target_suggests_importing_its_module() -> None:
    with pytest.raises(AglScopeError, match="Import"):
        _resolve_without_loader({"entry": "use Missing::*\n"})


def test_unrelated_nested_scope_does_not_mask_a_root_import_route(tmp_path: Path) -> None:
    """A nested ``X::lib`` scope must not shadow an unrelated root import named ``lib``."""
    resolution = _entry_resolution(
        tmp_path,
        {
            "entry": (
                "import lib\n"
                "\n"
                "scope X\n"
                "\n"
                "  scope lib\n"
                "    def g() -> int = 1\n"
                "  end lib\n"
                "end X\n"
                "\n"
                "print(lib::A::f())\n"
            ),
            "lib": "scope A\n  def f() -> int = 7\nend A\n",
        },
    )

    (call_ref,) = [node for node in _find_nodes(resolution.program, VarRef) if node.name == "f"]
    assert resolution.resolution[call_ref.node_id].module_id != ENTRY_ID


@pytest.mark.parametrize(
    "entry_source",
    (
        "import lib\nprint(lib[int]::A::f())",
        "import lib\nprint(lib::A[int]::f())",
    ),
)
def test_type_arguments_are_rejected_on_imported_route_and_scope_segments(
    tmp_path: Path, entry_source: str
) -> None:
    from tests.agl.ir_harness import make_graph_from_files

    with pytest.raises(AglScopeError, match="Type arguments cannot be applied"):
        resolve_program(
            make_graph_from_files(
                tmp_path, {"entry": entry_source, "lib": "scope A\n  def f() -> int = 7\nend A\n"}
            )
        )


def test_hidden_generic_does_not_validate_an_unrelated_plain_scope(tmp_path: Path) -> None:
    with pytest.raises(AglScopeError, match="Type arguments cannot be applied"):
        _entry_resolution(
            tmp_path,
            {
                "entry": (
                    "import one/shared\nimport two/shared hiding Box\nshared::Box[int]::describe()"
                ),
                "one/shared": "scope Box\n  def describe() -> int = 7\nend Box",
                "two/shared": "record Box[T]\n  value: T",
            },
        )


def test_type_arguments_on_an_imported_generic_type_scope_still_resolve(tmp_path: Path) -> None:
    """``lib::Box[int]::describe()`` keeps working: ``Box`` is a real generic type."""
    resolution = _entry_resolution(
        tmp_path,
        {
            "entry": "import lib\nprint(lib::Box[int]::describe())",
            "lib": 'record Box[T]\n  value: T\n\ndef Box::describe() -> text = "box"\n',
        },
    )
    (describe_call,) = [
        call
        for call in _find_nodes(resolution.program, Call)
        if isinstance(call.callee, VarRef) and call.callee.name == "describe"
    ]
    assert isinstance(describe_call.callee, VarRef)
    assert resolution.resolution[describe_call.callee.node_id].module_id != ENTRY_ID


def test_wildcard_import_tail_keeps_the_qualified_enum_owner_reachable() -> None:
    """``import lib::*`` contributes bare owners without narrowing routes."""
    span = SourceSpan(1, 1, 1, 1, 0, 0, UNKNOWN_SOURCE)
    module = ModuleId.from_path("lib")
    declaration = ImportDecl(
        module_path=module.segments,
        wildcard=False,
        alias=None,
        tail=(),
        hidden=(),
        span=span,
        node_id=1,
    )
    env = build_import_env(
        (declaration,),
        {declaration.node_id: SingleTarget(module)},
        {module: {"Color": (module, "Color"), ("Color", "Red"): (module, ("Color", "Red"))}},
    )

    assert env.unqualified["Color"] == frozenset({(module, "Color")})
    assert resolve_qualified(env, ("lib",), ("Color", "Red")) == QualResolutionFound(
        module, (module, ("Color", "Red"))
    )


def test_unanchored_qualifier_searches_enclosing_scopes_innermost_first() -> None:
    assert enclosing_scope_bases(("outer", "inner")) == (("outer", "inner"), ("outer",), ())


def test_module_rooted_qualifier_searches_the_module_root_only() -> None:
    assert enclosing_scope_bases(("outer", "inner"), rooted=True) == ((),)
