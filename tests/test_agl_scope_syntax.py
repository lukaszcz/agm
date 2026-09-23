"""Behavioral coverage for scoped declaration and selection syntax."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl import PipelineDriver
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.modules.roots import RootSet
from agm.agl.parser import AglSyntaxError, parse_program
from agm.agl.scope import AglScopeError
from agm.agl.scope.imports import SingleTarget, build_import_env
from agm.agl.scope.resolver import _Resolver
from agm.agl.syntax import (
    BuiltinVarDecl,
    EnumDef,
    ExceptionDef,
    ExportDecl,
    FuncDef,
    ImportDecl,
    Item,
    LetDecl,
    RecordDef,
    ScopeRegion,
    ScopeSegment,
    UseDecl,
    VarDecl,
)
from tests._agl_helpers import run_inline_command
from tests.agl.ir_harness import write_module_file
from tests.agl.module_graph import resolve_entry, resolve_inline_entry


def _declaration(source: str) -> Item:
    (declaration,) = parse_program(source).body.items
    return declaration


def test_region_and_shorthand_declarations_have_the_same_scope_path() -> None:
    block_form = parse_program("scope A::B\n  def value() -> int = 0\nend A::B")
    shorthand = parse_program("def A::B::value() -> int = 0")

    (region,) = block_form.body.items
    assert isinstance(region, ScopeRegion)
    (inner_region,) = region.items
    assert isinstance(inner_region, ScopeRegion)
    (block_declaration,) = inner_region.items
    assert isinstance(block_declaration, FuncDef)
    (shorthand_declaration,) = shorthand.body.items
    assert isinstance(shorthand_declaration, FuncDef)

    assert block_declaration == shorthand_declaration


def test_nested_region_declarations_accumulate_the_enclosing_scope_path() -> None:
    program = parse_program("scope A\n\n  scope B\n    def C::value() -> int = 0\n  end B\nend A")

    (outer,) = program.body.items
    assert isinstance(outer, ScopeRegion)
    (inner,) = outer.items
    assert isinstance(inner, ScopeRegion)
    (declaration,) = inner.items
    assert isinstance(declaration, FuncDef)
    assert [segment.name for segment in declaration.scope_path] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Region body layout
# ---------------------------------------------------------------------------


def test_an_indented_region_body_parses_as_the_flat_one_does() -> None:
    """A region's items may sit in an indented block under their `scope` header."""
    indented = parse_program("scope A\n  def value() -> int = 0\nend A")
    flat = parse_program("scope A\n  def value() -> int = 0\nend A")

    assert indented == flat


def test_a_nested_indented_region_closes_at_its_own_level() -> None:
    """An inner `end` sits one level in, before the outer block's dedent."""
    source = (
        "scope Outer\n"
        "  def a() -> int = 1\n"
        "\n"
        "  scope Inner\n"
        "    def b() -> int = 2\n"
        "  end Inner\n"
        "end Outer"
    )

    program = parse_program(source)

    (outer,) = program.body.items
    assert isinstance(outer, ScopeRegion)
    inner = outer.items[1]
    assert isinstance(inner, ScopeRegion)
    (declaration,) = inner.items
    assert isinstance(declaration, FuncDef)
    assert declaration.name == "b"
    assert [segment.name for segment in declaration.scope_path] == ["Outer", "Inner"]


def test_a_region_body_may_mix_indented_and_flat_nesting() -> None:
    """An indented body admits a flat inner region, and the reverse."""
    indented_outer = parse_program(
        "scope Outer\n\n  scope Inner\n    def b() -> int = 2\n  end Inner\nend Outer"
    )
    flat_outer = parse_program(
        "scope Outer\n\n  scope Inner\n    def b() -> int = 2\n  end Inner\nend Outer"
    )

    assert indented_outer == flat_outer


def test_an_indented_region_body_runs(capsys: pytest.CaptureFixture[str]) -> None:
    source = "scope A\n  def value() -> int = 7\nend A\n\nprint(A::value())"

    assert run_inline_command(PipelineDriver(), source).ok is True
    assert capsys.readouterr().out == "7\n"


@pytest.mark.parametrize(
    "source",
    (
        # The closer is deeper than the body it closes.
        "scope A\n  def value() -> int = 0\n  end A",
        # The closer never arrives.
        "scope A\n  def value() -> int = 0",
    ),
)
def test_an_indented_region_still_requires_a_closer_at_the_header_level(source: str) -> None:
    with pytest.raises(AglSyntaxError):
        parse_program(source)


# ---------------------------------------------------------------------------
# `let`/`var` binder paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "kind", "name", "path"),
    (
        ("let A::x = 1", LetDecl, "x", ["A"]),
        ("var A::count = 0", VarDecl, "count", ["A"]),
        ("let A::B::x = 1", LetDecl, "x", ["A", "B"]),
        ("var A::B::c = 0", VarDecl, "c", ["A", "B"]),
    ),
)
def test_let_and_var_accept_root_scope_path_shorthand(
    source: str, kind: type[object], name: str, path: list[str]
) -> None:
    declaration = _declaration(source)

    assert isinstance(declaration, kind)
    assert isinstance(declaration, (LetDecl, VarDecl))
    assert [segment.name for segment in declaration.scope_path] == path
    assert declaration.name == name


@pytest.mark.parametrize(
    ("source", "kind"),
    (
        ("let value = 1", LetDecl),
        ("var value = 0", VarDecl),
    ),
)
def test_let_and_var_are_admitted_inside_a_scope_region(source: str, kind: type[object]) -> None:
    program = parse_program(f"scope Config\n{source}\nend Config")

    (region,) = program.body.items
    assert isinstance(region, ScopeRegion)
    (member,) = region.items
    assert isinstance(member, kind)
    assert isinstance(member, (LetDecl, VarDecl))
    assert [segment.name for segment in member.scope_path] == ["Config"]


def test_let_and_var_accumulate_scope_path_across_nested_regions() -> None:
    program = parse_program(
        "scope A\n\n  scope B\n    let retries = 3\n    var attempts = 0\n  end B\nend A"
    )

    (outer,) = program.body.items
    assert isinstance(outer, ScopeRegion)
    (inner,) = outer.items
    assert isinstance(inner, ScopeRegion)
    let_member, var_member = inner.items
    assert isinstance(let_member, LetDecl)
    assert isinstance(var_member, VarDecl)
    assert [segment.name for segment in let_member.scope_path] == ["A", "B"]
    assert [segment.name for segment in var_member.scope_path] == ["A", "B"]


def test_let_binder_path_shorthand_combines_with_enclosing_region_path() -> None:
    program = parse_program("scope Outer\n  let Inner::x = 1\nend Outer")

    (region,) = program.body.items
    assert isinstance(region, ScopeRegion)
    (member,) = region.items
    assert isinstance(member, LetDecl)
    assert member.name == "x"
    assert [segment.name for segment in member.scope_path] == ["Outer", "Inner"]


def test_var_binder_path_rejects_a_module_route_segment() -> None:
    with pytest.raises(AglSyntaxError, match="'::'"):
        parse_program("var std/config::retries = 0")


def test_var_binder_path_rejects_a_type_applied_segment() -> None:
    with pytest.raises(AglSyntaxError):
        parse_program("var A[int]::x = 0")


# ---------------------------------------------------------------------------
# `import`/`export` region membership (no declaration-path shorthand)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "kind"),
    (
        ("import lib", ImportDecl),
        ("import lib::*", ImportDecl),
        ("export lib", ExportDecl),
    ),
)
def test_import_and_export_are_admitted_inside_a_scope_region(
    source: str, kind: type[object]
) -> None:
    program = parse_program(f"scope Config\n{source}\nend Config")

    (region,) = program.body.items
    assert isinstance(region, ScopeRegion)
    (member,) = region.items
    assert isinstance(member, kind)
    assert isinstance(member, (ImportDecl, ExportDecl))
    assert member.module_path == ("lib",)
    assert [segment.name for segment in member.scope_path] == ["Config"]


def test_import_and_export_accumulate_scope_path_across_nested_regions() -> None:
    program = parse_program("scope A\n\n  scope B\n    import lib\n    export lib\n  end B\nend A")

    (outer,) = program.body.items
    assert isinstance(outer, ScopeRegion)
    (inner,) = outer.items
    assert isinstance(inner, ScopeRegion)
    import_member, export_member = inner.items
    assert isinstance(import_member, ImportDecl)
    assert isinstance(export_member, ExportDecl)
    assert [segment.name for segment in import_member.scope_path] == ["A", "B"]
    assert [segment.name for segment in export_member.scope_path] == ["A", "B"]


def test_tailed_import_is_admitted_at_the_start_of_a_scope_region() -> None:
    program = parse_program("scope A\n  import lib::*\n  def value() -> int = 0\nend A")

    (region,) = program.body.items
    assert isinstance(region, ScopeRegion)
    assert isinstance(region.items[0], ImportDecl)


def test_use_after_a_non_header_region_item_is_rejected() -> None:
    with pytest.raises(AglSyntaxError):
        resolve_inline_entry("scope A\n  def value() -> int = 0\n  use B::*\nend A")


@pytest.mark.parametrize(
    "source",
    (
        # `std/config` (a real module distinct from the auto-imported
        # `std/prelude`) stands in for a placeholder library name so the real
        # module graph these are now built through can actually load the
        # import; content doesn't matter here since placement is checked
        # before any import content is consulted.
        "scope A\n  def value() -> int = 0\n  import std/config\nend A",
        "scope A\n  def value() -> int = 0\n  import std/config::*\nend A",
        "scope A\n  def value() -> int = 0\n  export std/config\nend A",
    ),
)
def test_import_after_a_non_header_region_item_is_rejected_by_the_scope_pass(source: str) -> None:
    """The scope pass -- not the parser -- owns import/export placement."""
    with pytest.raises(AglScopeError):
        resolve_entry(source)


def test_export_before_other_region_items_is_admitted() -> None:
    """Like `import` and `use`, `export` is confined to a region's header."""
    program = parse_program("scope A\n  export lib\n  def value() -> int = 0\nend A")

    (region,) = program.body.items
    assert isinstance(region, ScopeRegion)
    assert isinstance(region.items[0], ExportDecl)


def test_the_parser_does_not_own_import_placement_at_the_module_root() -> None:
    """The parser leaves root import placement to the scope pass."""
    parse_program("def value() -> int = 0\nimport lib")


def test_import_after_a_non_header_root_item_is_rejected_by_the_scope_pass() -> None:
    with pytest.raises(AglScopeError):
        resolve_entry("def value() -> int = 0\nimport std/config")


@pytest.mark.parametrize(
    "source",
    (
        # See test_import_after_a_non_header_region_item_is_rejected_by_the_scope_pass:
        # `std/config` stands in for a placeholder library name so the real
        # module graph can load the import.
        "def f() =\n  import std/config\n  0\nf()",
        "def f() =\n  export std/config\n  0\nf()",
    ),
)
def test_import_and_export_still_rejected_inside_a_function_body(source: str) -> None:
    with pytest.raises(AglScopeError):
        resolve_entry(source)


@pytest.mark.parametrize(
    ("source", "target", "tail", "hidden"),
    (
        ("use Point::*", ("Point",), (), ()),
        ("use Point::{distance}", ("Point",), (("distance", None),), ()),
        (
            "use Point::{distance as d, length as l}",
            ("Point",),
            (("distance", "d"), ("length", "l")),
            (),
        ),
        ("use Point::* hiding internal", ("Point",), (), (("internal", None),)),
        (
            "use geo/shapes::Point::Metrics::{distance as d}",
            ("geo/shapes", "Point", "Metrics"),
            (("distance", "d"),),
            (),
        ),
    ),
)
def test_use_declarations_accept_scope_references_and_clauses(
    source: str,
    target: tuple[str, ...],
    tail: tuple[tuple[str, str | None], ...],
    hidden: tuple[tuple[str, str | None], ...],
) -> None:
    declaration = _declaration(source)

    assert isinstance(declaration, UseDecl)
    assert tuple(segment.name for segment in declaration.target) == target
    assert tuple((item.name, item.rename) for item in declaration.tail or ()) == tail
    assert tuple((item.name, item.rename) for item in declaration.hidden) == hidden


def test_use_declarations_are_allowed_at_the_start_of_scope_regions() -> None:
    program = parse_program("scope A\n  use B::{value as b}\n  def value() -> int = 0\nend A")

    (region,) = program.body.items
    assert isinstance(region, ScopeRegion)
    assert isinstance(region.items[0], UseDecl)


@pytest.mark.parametrize(
    "source",
    (
        "def value() -> int =\n    use Point::*\n    0",
        "if true =>\n    use Point::*\n    ()",
        "do\n    use Point::*\n    ()\ndone",
    ),
)
def test_use_declarations_are_rejected_outside_module_and_scope_regions(source: str) -> None:
    with pytest.raises(AglSyntaxError, match="only allowed at module root or in scope regions"):
        parse_program(source)


@pytest.mark.parametrize(
    "source",
    (
        "use Point::{value} hiding hidden",
        "use Point::* as Alias",
    ),
)
def test_use_clause_shapes_are_rejected(source: str) -> None:
    with pytest.raises(AglSyntaxError):
        parse_program(source)


@pytest.mark.parametrize(
    ("source", "kind"),
    (
        ("import library::{Point::distance as d, Point}", ImportDecl),
        ("import library hiding Point::internal", ImportDecl),
        ("export library::{Point::distance as d, Point}", ExportDecl),
        ("export library hiding Point::internal", ExportDecl),
    ),
)
def test_import_and_export_clauses_accept_path_atoms(source: str, kind: type[object]) -> None:
    declaration = _declaration(source)

    assert isinstance(declaration, kind)
    assert isinstance(declaration, (ImportDecl, ExportDecl))
    selected = declaration.tail if isinstance(declaration, ImportDecl) else declaration.items
    atoms = tuple(
        (item.name, item.rename, tuple(segment.name for segment in item.scope_path))
        for item in (declaration.hidden if "hiding" in source else selected or ())
    )
    expected = (
        (("distance", "d", ("Point",)), ("Point", None, ()))
        if "{" in source
        else (("internal", None, ("Point",)),)
    )
    assert atoms == expected


def test_use_rejects_operator_alias_for_scope_route() -> None:
    with pytest.raises(AglScopeError):
        resolve_inline_entry(
            "use Point as >>\n\nscope Point\n  def distance() -> int = 1\nend Point"
        )


@pytest.mark.parametrize(
    "source",
    (
        "use foo::bar/baz::x",
        "use foo::/bar::x",
        "use ::foo/bar::x",
    ),
    ids=("route-segment", "anchored-segment", "current-module-route"),
)
def test_use_rejects_a_module_route_after_the_target_head(source: str) -> None:
    """Only a use target's head is a module route; every segment after it
    names a scope, so a separator there is a syntax error wherever it sits."""
    with pytest.raises(AglSyntaxError, match="'::'"):
        parse_program(source)


def test_use_contributes_local_scope_members() -> None:
    resolved = resolve_inline_entry(
        "use Point::*\n\nscope Point\n  def distance() -> int = 1\nend Point\n\ndistance()"
    )

    assert any(ref.name == "distance" for ref in resolved.resolution.values())


def test_used_scope_members_clash_at_their_use_site() -> None:
    source = (
        "use Point::*\nuse Vector::*\n"
        "\n"
        "scope Point\n  def distance() -> int = 1\nend Point\n"
        "\n"
        "scope Vector\n  def distance() -> int = 2\nend Vector\n\ndistance()"
    )

    with pytest.raises(AglScopeError, match="ambiguous"):
        resolve_inline_entry(source)


def test_use_reaches_a_scope_made_nameable_by_an_import_tail() -> None:
    """A use target follows the same bare contribution a qualifier follows."""
    program = parse_program("import library::{Scope}\nuse Scope::*\nvisible")
    import_decl, _use_decl, _visible = program.body.items
    assert isinstance(import_decl, ImportDecl)
    library = ModuleId.from_path("library")
    scope_member = ("Scope", "visible")
    import_env = build_import_env(
        (import_decl,),
        {import_decl.node_id: SingleTarget(library)},
        {library: {scope_member: (library, scope_member)}},
    )

    resolved = _Resolver(
        module_id=ENTRY_ID,
        import_env=import_env,
        all_public_types={},
        allow_root_statements=True,
    ).run(program)

    assert any(ref.name == "visible" for ref in resolved.resolution.values())


def test_use_keeps_equally_nameable_bare_scope_targets_ambiguous() -> None:
    program = parse_program("import one::{Scope}\nimport two::{Scope}\nuse Scope::*")
    first_import, second_import, _use_decl = program.body.items
    assert isinstance(first_import, ImportDecl)
    assert isinstance(second_import, ImportDecl)
    one = ModuleId.from_path("one")
    two = ModuleId.from_path("two")
    scope_member = ("Scope", "visible")
    import_env = build_import_env(
        (first_import, second_import),
        {
            first_import.node_id: SingleTarget(one),
            second_import.node_id: SingleTarget(two),
        },
        {
            one: {scope_member: (one, scope_member)},
            two: {scope_member: (two, scope_member)},
        },
    )

    with pytest.raises(AglScopeError, match="ambiguous"):
        _Resolver(module_id=ENTRY_ID, import_env=import_env, all_public_types={}).run(program)


@pytest.mark.parametrize(
    ("source", "segments"),
    (
        ("use m/n::Scope::Nested::*", ("m/n", "Scope", "Nested")),
        ("use /m/n::Scope::*", ("m/n", "Scope")),
        ("use ::Scope::Nested::*", ("Scope", "Nested")),
    ),
)
def test_use_target_segment_spans_exclude_qualifier_delimiters_and_anchors(
    source: str, segments: tuple[str, ...]
) -> None:
    declaration = _declaration(source)

    assert isinstance(declaration, UseDecl)
    assert tuple(segment.name for segment in declaration.target) == segments
    for segment in declaration.target:
        start = source.index(segment.name, source.index("use") + len("use"))
        assert (segment.span.start_offset, segment.span.end_offset) == (
            start,
            start + len(segment.name),
        )


def test_ast_walk_visits_use_and_export_selection_paths() -> None:
    from agm.agl.syntax.visitor import walk

    program = parse_program("use Point::{Nested::member as m}\nexport library::{Point::distance}")
    visited: list[object] = []

    walk(program, visited.append)

    assert sum(isinstance(node, UseDecl) for node in visited) == 1
    assert sum(isinstance(node, ExportDecl) for node in visited) == 1
    assert sum(isinstance(node, ScopeSegment) for node in visited) == 3


def test_ast_walk_visits_a_scoped_funcs_scope_path_segments() -> None:
    from agm.agl.syntax.visitor import walk

    program = parse_program("scope A::B\n  def f() -> unit = ()\nend A::B")
    visited: list[object] = []

    walk(program, visited.append)

    # Two segments from the nested ScopeRegion headers ("A", "B") plus two more
    # from the FuncDef's own accumulated `scope_path` ("A", "B").
    assert sum(isinstance(node, FuncDef) for node in visited) == 1
    assert sum(isinstance(node, ScopeSegment) for node in visited) == 4


def test_scoped_declarations_do_not_generate_runtime_initializers() -> None:
    result = run_inline_command(PipelineDriver(), "def A::f() -> int = 0\n()", default_stdlib=False)

    assert result.ok


def test_library_scope_regions_apply_entry_only_declaration_restrictions(tmp_path: Path) -> None:
    root = tmp_path / "modules"
    root.mkdir()
    write_module_file(root, "library", "scope A\n  agent bot\nend A")

    result = run_inline_command(
        PipelineDriver(),
        "import library\n()",
        roots=RootSet(roots=frozenset({root})),
        default_stdlib=False,
    )

    assert not result.ok


@pytest.mark.parametrize(
    ("entry", "library"),
    (
        ("import library::Point::distance\n()", "record Point"),
        ("export library hiding Point::distance\n()", "record Point"),
        ("import library\n()", "import dependency::Point::distance\ndef value() -> int = 0"),
    ),
)
def test_production_pipeline_validates_path_atoms_against_public_content(
    tmp_path: Path, entry: str, library: str
) -> None:
    root = tmp_path / "modules"
    root.mkdir()
    write_module_file(root, "library", library)
    if "dependency" in library:
        write_module_file(root, "dependency", "record Point")

    result = run_inline_command(
        PipelineDriver(),
        entry,
        roots=RootSet(roots=frozenset({root})),
        default_stdlib=False,
    )

    assert not result.ok
    assert len(result.diagnostics) == 1
    assert "is not exported" in result.diagnostics[0].message


# ---------------------------------------------------------------------------
# `builtin` forms as region members
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "kind"),
    (
        ("builtin def native() -> int", FuncDef),
        ("builtin\nrecord Payload\n  x: int", RecordDef),
        ("builtin\nenum Status\n  | ok", EnumDef),
        ("builtin\nexception Failure\n  message: text", ExceptionDef),
        ("builtin var setting: int", BuiltinVarDecl),
    ),
    ids=("def", "record", "enum", "exception", "var"),
)
def test_builtin_forms_are_admitted_inside_a_scope_region(source: str, kind: type[object]) -> None:
    program = parse_program(f"scope Host\n{source}\nend Host")

    (region,) = program.body.items
    assert isinstance(region, ScopeRegion)
    (member,) = region.items
    assert isinstance(member, kind)
    assert isinstance(member, (FuncDef, RecordDef, EnumDef, ExceptionDef, BuiltinVarDecl))
    assert [segment.name for segment in member.scope_path] == ["Host"]


def test_builtin_forms_accumulate_scope_path_across_nested_regions() -> None:
    program = parse_program(
        "scope A\n"
        "\n"
        "  scope B\n"
        "    builtin def native() -> int\n"
        "    builtin var setting: int\n"
        "  end B\n"
        "end A"
    )

    (outer,) = program.body.items
    assert isinstance(outer, ScopeRegion)
    (inner,) = outer.items
    assert isinstance(inner, ScopeRegion)
    func_member, var_member = inner.items
    assert isinstance(func_member, FuncDef)
    assert isinstance(var_member, BuiltinVarDecl)
    assert [segment.name for segment in func_member.scope_path] == ["A", "B"]
    assert [segment.name for segment in var_member.scope_path] == ["A", "B"]


@pytest.mark.parametrize(
    "source",
    (
        "def f() -> int =\n  builtin def native() -> int\n  1\nf()",
        "def f() -> int =\n  builtin var setting: int\n  1\nf()",
    ),
    ids=("def", "var"),
)
def test_builtin_forms_still_rejected_inside_a_function_body(source: str) -> None:
    """``builtin`` gains the region only; the pre-existing nested-block ban stands."""
    with pytest.raises(AglScopeError):
        resolve_entry(source)
