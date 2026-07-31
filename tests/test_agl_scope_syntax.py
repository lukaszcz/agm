"""Behavioral coverage for scoped declaration and selection syntax."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl import PipelineDriver
from agm.agl.modules.roots import RootSet
from agm.agl.parser import AglSyntaxError, parse_program
from agm.agl.scope import AglScopeError, resolve_module
from agm.agl.scope.symbols import resolve_bare_contribution
from agm.agl.syntax import (
    AgentDecl,
    AsPattern,
    ConstructorPattern,
    EnumDef,
    ExceptionDef,
    ExportDecl,
    FuncDef,
    ImportDecl,
    LetDecl,
    OpenDecl,
    ParamDecl,
    RecordDef,
    ScopeRegion,
    ScopeSegment,
    TypeAlias,
    VarDecl,
    VarPattern,
)
from tests.agl.ir_harness import write_module_file


def _declaration(source: str) -> object:
    (declaration,) = parse_program(source).body.items
    return declaration


@pytest.mark.parametrize(
    ("source", "kind"),
    (
        ("def A::B::value() -> int = 0", FuncDef),
        ("extern def A::B::value() -> int", FuncDef),
        ("record A::B::Point(x: int)", RecordDef),
        ("enum A::B::Result = ok", EnumDef),
        ("exception A::B::Failure(message: text)", ExceptionDef),
        ("type A::B::Count = int", TypeAlias),
        ('agent A::B::reviewer = "runner"', AgentDecl),
    ),
)
def test_name_headed_declarations_accept_scope_path_shorthand(
    source: str, kind: type[object]
) -> None:
    declaration = _declaration(source)

    assert isinstance(declaration, kind)
    assert declaration.name in {"value", "Point", "Result", "Failure", "Count", "reviewer"}
    assert [segment.name for segment in declaration.scope_path] == ["A", "B"]
    resolve_module(parse_program(source), origin_path=Path("module.agl"))


@pytest.mark.parametrize(
    ("source", "kind", "name"),
    (
        ("def Inner::value() -> int = 0", FuncDef, "value"),
        ("extern def Inner::value() -> int", FuncDef, "value"),
        ("record Inner::Point(x: int)", RecordDef, "Point"),
        ("enum Inner::Result = ok", EnumDef, "Result"),
        ("exception Inner::Failure(message: text)", ExceptionDef, "Failure"),
        ("type Inner::Count = int", TypeAlias, "Count"),
        ('agent Inner::reviewer = "runner"', AgentDecl, "reviewer"),
    ),
)
def test_scope_region_declarations_accumulate_shorthand_scope_paths(
    source: str, kind: type[object], name: str
) -> None:
    program = parse_program(f"scope Outer\n{source}\nend Outer")

    (region,) = program.body.items
    assert isinstance(region, ScopeRegion)
    (declaration,) = region.items
    assert isinstance(declaration, kind)
    assert declaration.name == name
    assert [segment.name for segment in declaration.scope_path] == ["Outer", "Inner"]


def test_region_and_shorthand_declarations_have_the_same_scope_path() -> None:
    block_form = parse_program("scope A::B\ndef value() -> int = 0\nend A::B")
    shorthand = parse_program("def A::B::value() -> int = 0")

    (region,) = block_form.body.items
    assert isinstance(region, ScopeRegion)
    (block_declaration,) = region.items[0].items
    assert isinstance(block_declaration, FuncDef)
    (shorthand_declaration,) = shorthand.body.items
    assert isinstance(shorthand_declaration, FuncDef)

    assert block_declaration == shorthand_declaration


def test_nested_region_declarations_accumulate_the_enclosing_scope_path() -> None:
    program = parse_program("scope A\nscope B\ndef C::value() -> int = 0\nend B\nend A")

    (outer,) = program.body.items
    assert isinstance(outer, ScopeRegion)
    (inner,) = outer.items
    assert isinstance(inner, ScopeRegion)
    (declaration,) = inner.items
    assert isinstance(declaration, FuncDef)
    assert [segment.name for segment in declaration.scope_path] == ["A", "B", "C"]


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
    assert [segment.name for segment in declaration.scope_path] == path
    if isinstance(declaration, LetDecl):
        assert isinstance(declaration.pattern, VarPattern)
        assert declaration.pattern.name == name
    else:
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
    assert [segment.name for segment in member.scope_path] == ["Config"]


def test_let_and_var_accumulate_scope_path_across_nested_regions() -> None:
    program = parse_program("scope A\nscope B\nlet retries = 3\nvar attempts = 0\nend B\nend A")

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
    program = parse_program("scope Outer\nlet Inner::x = 1\nend Outer")

    (region,) = program.body.items
    assert isinstance(region, ScopeRegion)
    (member,) = region.items
    assert isinstance(member, LetDecl)
    assert isinstance(member.pattern, VarPattern)
    assert member.pattern.name == "x"
    assert [segment.name for segment in member.scope_path] == ["Outer", "Inner"]


@pytest.mark.parametrize(
    ("source", "pattern_kind", "scope_path"),
    (
        # `let A::x = e` — a chain spellable as a declaration head reinterprets
        # as a scoped binding.
        ("let A::x = e", VarPattern, ("A",)),
        # `let A::x() = e` — an explicit (empty) argument list forces a
        # nullary qualified constructor pattern.
        ("let A::x() = e", ConstructorPattern, ()),
        # `let A::x(a, b) = e` — a constructor pattern with fields.
        ("let A::x(a, b) = e", ConstructorPattern, ()),
        # `let A::x as y = e` — an `as` binder always wraps a match pattern.
        ("let A::x as y = e", AsPattern, ()),
        # `let x = e` — an unqualified name is an ordinary root binding.
        ("let x = e", VarPattern, ()),
        # A `::`-anchored chain is not spellable as a declaration head, so it
        # keeps its constructor-pattern meaning.
        ("let ::x = e", ConstructorPattern, ()),
        ("let ::A::x = e", ConstructorPattern, ()),
        # A module-routed segment is not spellable as a declaration head
        # either, so it too keeps its constructor-pattern meaning.
        ("let std/config::retries = e", ConstructorPattern, ()),
        ("let config::A/B::retries = e", ConstructorPattern, ()),
        # A type-argument-applied segment likewise keeps its pattern meaning.
        ("let Box[int]::v = e", ConstructorPattern, ()),
    ),
)
def test_let_disambiguation_table(
    source: str, pattern_kind: type[object], scope_path: tuple[str, ...]
) -> None:
    declaration = _declaration(source)

    assert isinstance(declaration, LetDecl)
    assert isinstance(declaration.pattern, pattern_kind)
    assert [segment.name for segment in declaration.scope_path] == list(scope_path)


def test_let_as_binder_is_never_reinterpreted_as_a_scoped_binding() -> None:
    declaration = _declaration("let A::x as y = e")

    assert isinstance(declaration, LetDecl)
    assert isinstance(declaration.pattern, AsPattern)
    assert declaration.pattern.name == "y"
    assert declaration.scope_path == ()


def test_var_binder_path_rejects_a_module_route_segment() -> None:
    with pytest.raises(AglSyntaxError, match="'::'"):
        parse_program("var std/config::retries = 0")


def test_var_binder_path_rejects_a_type_applied_segment() -> None:
    with pytest.raises(AglSyntaxError):
        parse_program("var A[int]::x = 0")


# ---------------------------------------------------------------------------
# `param` region membership (no declaration-path shorthand)
# ---------------------------------------------------------------------------


def test_param_is_admitted_inside_a_scope_region() -> None:
    program = parse_program("scope Deploy\nparam region: text\nend Deploy")

    (region,) = program.body.items
    assert isinstance(region, ScopeRegion)
    (member,) = region.items
    assert isinstance(member, ParamDecl)
    assert member.name == "region"
    assert [segment.name for segment in member.scope_path] == ["Deploy"]


def test_param_accumulates_scope_path_across_nested_regions() -> None:
    program = parse_program("scope A\nscope B\nparam x\nend B\nend A")

    (outer,) = program.body.items
    assert isinstance(outer, ScopeRegion)
    (inner,) = outer.items
    assert isinstance(inner, ScopeRegion)
    (member,) = inner.items
    assert isinstance(member, ParamDecl)
    assert [segment.name for segment in member.scope_path] == ["A", "B"]


def test_param_accepts_a_multi_segment_region_header() -> None:
    program = parse_program("scope A::B\nparam x\nend A::B")

    (outer,) = program.body.items
    assert isinstance(outer, ScopeRegion)
    (inner,) = outer.items
    assert isinstance(inner, ScopeRegion)
    (member,) = inner.items
    assert isinstance(member, ParamDecl)
    assert [segment.name for segment in member.scope_path] == ["A", "B"]


def test_param_has_no_declaration_path_shorthand() -> None:
    """Unlike `def`/`let`/`var`, `param` has only the region spelling."""
    with pytest.raises(AglSyntaxError):
        parse_program("param A::x")


def test_param_still_rejected_inside_a_function_body() -> None:
    with pytest.raises(AglScopeError, match="param"):
        resolve_module(parse_program("def f() =\n  param x\n  0\nf()"))


# ---------------------------------------------------------------------------
# `import`/`export` region membership (no declaration-path shorthand)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "kind"),
    (
        ("import lib", ImportDecl),
        ("open import lib", ImportDecl),
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
    assert member.module_path == ("lib",)
    assert [segment.name for segment in member.scope_path] == ["Config"]


def test_import_and_export_accumulate_scope_path_across_nested_regions() -> None:
    program = parse_program("scope A\nscope B\nimport lib\nexport lib\nend B\nend A")

    (outer,) = program.body.items
    assert isinstance(outer, ScopeRegion)
    (inner,) = outer.items
    assert isinstance(inner, ScopeRegion)
    import_member, export_member = inner.items
    assert isinstance(import_member, ImportDecl)
    assert isinstance(export_member, ExportDecl)
    assert [segment.name for segment in import_member.scope_path] == ["A", "B"]
    assert [segment.name for segment in export_member.scope_path] == ["A", "B"]


def test_open_import_is_admitted_at_the_start_of_a_scope_region() -> None:
    program = parse_program("scope A\nopen import lib\ndef value() -> int = 0\nend A")

    (region,) = program.body.items
    assert isinstance(region, ScopeRegion)
    assert isinstance(region.items[0], ImportDecl)


@pytest.mark.parametrize(
    "source",
    (
        "scope A\ndef value() -> int = 0\nimport lib\nend A",
        "scope A\ndef value() -> int = 0\nopen import lib\nend A",
        "scope A\ndef value() -> int = 0\nexport lib\nend A",
    ),
)
def test_import_after_a_non_header_region_item_is_rejected(source: str) -> None:
    with pytest.raises(AglSyntaxError):
        parse_program(source)


def test_export_before_other_region_items_is_admitted() -> None:
    """Like `import`/`open`, `export` is confined to a region's header."""
    program = parse_program("scope A\nexport lib\ndef value() -> int = 0\nend A")

    (region,) = program.body.items
    assert isinstance(region, ScopeRegion)
    assert isinstance(region.items[0], ExportDecl)


def test_import_placement_at_the_module_root_is_unaffected_by_the_region_header_rule() -> None:
    """The region-only header check must not leak into ordinary root placement."""
    parse_program("def value() -> int = 0\nimport lib")


@pytest.mark.parametrize(
    "source",
    (
        "def f() =\n  import lib\n  0\nf()",
        "def f() =\n  export lib\n  0\nf()",
    ),
)
def test_import_and_export_still_rejected_inside_a_function_body(source: str) -> None:
    with pytest.raises(AglScopeError):
        resolve_module(parse_program(source))


@pytest.mark.parametrize(
    ("source", "module_route", "scope_path", "mode", "items"),
    (
        ("open Point", (), ("Point",), "all", ()),
        ("open Point using distance", (), ("Point",), "using", (("distance", None),)),
        (
            "open Point using distance as d, length as l",
            (),
            ("Point",),
            "using",
            (("distance", "d"), ("length", "l")),
        ),
        ("open Point hiding internal", (), ("Point",), "hiding", (("internal", None),)),
        (
            "open geo/shapes::Point::Metrics using distance as d",
            ("geo", "shapes"),
            ("Point", "Metrics"),
            "using",
            (("distance", "d"),),
        ),
    ),
)
def test_open_declarations_accept_scope_references_and_clauses(
    source: str,
    module_route: tuple[str, ...],
    scope_path: tuple[str, ...],
    mode: str,
    items: tuple[tuple[str, str | None], ...],
) -> None:
    declaration = _declaration(source)

    assert isinstance(declaration, OpenDecl)
    assert declaration.scope_ref.module_route == module_route
    assert tuple(segment.name for segment in declaration.scope_ref.scope_path) == scope_path
    assert declaration.mode.name.lower() == mode
    assert tuple((item.name, item.rename) for item in declaration.items) == items


def test_open_declarations_are_allowed_at_the_start_of_scope_regions() -> None:
    program = parse_program("scope A\nopen B using value as b\ndef value() -> int = 0\nend A")

    (region,) = program.body.items
    assert isinstance(region, ScopeRegion)
    assert isinstance(region.items[0], OpenDecl)


@pytest.mark.parametrize(
    "source",
    (
        "def value() -> int =\n    open Point\n    0",
        "if true =>\n    open Point\n    ()",
        "do\n    open Point\n    ()\ndone",
    ),
)
def test_open_declarations_are_rejected_outside_module_and_scope_regions(source: str) -> None:
    with pytest.raises(AglSyntaxError, match="only allowed at module root or in scope regions"):
        parse_program(source)


@pytest.mark.parametrize(
    "source",
    (
        "open Point using value hiding hidden",
        "def value() -> int = 0\nopen Point",
        "scope A\ndef value() -> int = 0\nopen B\nend A",
    ),
)
def test_open_clause_shapes_and_placement_are_rejected(source: str) -> None:
    with pytest.raises(AglSyntaxError):
        parse_program(source)


@pytest.mark.parametrize(
    ("source", "kind"),
    (
        ("import library using Point::distance as d, Point", ImportDecl),
        ("import library hiding Point::internal", ImportDecl),
        ("export library using Point::distance as d, Point", ExportDecl),
        ("export library hiding Point::internal", ExportDecl),
    ),
)
def test_import_and_export_clauses_accept_path_atoms(source: str, kind: type[object]) -> None:
    declaration = _declaration(source)

    assert isinstance(declaration, kind)
    atoms = tuple(
        (item.name, item.rename, tuple(segment.name for segment in item.scope_path))
        for item in declaration.items
    )
    expected = (
        (("distance", "d", ("Point",)), ("Point", None, ()))
        if "using" in source
        else (("internal", None, ("Point",)),)
    )
    assert atoms == expected


def test_scope_pass_opens_local_scope_members() -> None:
    resolved = resolve_module(
        parse_program("open Point\nscope Point\ndef distance() -> int = 1\nend Point\ndistance()")
    )

    assert (
        len(resolve_bare_contribution(resolved.root_scope, "distance", resolved.scope_nodes) or ())
        == 1
    )


def test_opened_scope_members_clash_at_their_use_site() -> None:
    source = (
        "open Point\nopen Vector\n"
        "scope Point\ndef distance() -> int = 1\nend Point\n"
        "scope Vector\ndef distance() -> int = 2\nend Vector\ndistance()"
    )

    with pytest.raises(AglScopeError, match="ambiguous"):
        resolve_module(parse_program(source))


def test_ast_walk_visits_open_and_export_selection_paths() -> None:
    from agm.agl.syntax.visitor import walk

    program = parse_program(
        "open Point using Nested::member as m\nexport library using Point::distance"
    )
    visited: list[object] = []

    walk(program, visited.append)

    assert sum(isinstance(node, OpenDecl) for node in visited) == 1
    assert sum(isinstance(node, ExportDecl) for node in visited) == 1
    assert sum(isinstance(node, ScopeSegment) for node in visited) == 3


def test_ast_walk_visits_a_scoped_params_scope_path_segments() -> None:
    from agm.agl.syntax.visitor import walk

    program = parse_program("scope A::B\nparam x\nend A::B")
    visited: list[object] = []

    walk(program, visited.append)

    # Two segments from the nested ScopeRegion headers ("A", "B") plus two more
    # from the ParamDecl's own accumulated `scope_path` ("A", "B").
    assert sum(isinstance(node, ParamDecl) for node in visited) == 1
    assert sum(isinstance(node, ScopeSegment) for node in visited) == 4


@pytest.mark.parametrize(
    "source",
    (
        "import library/* using Point::distance",
        "export library/* hiding Point::internal",
    ),
)
def test_scope_pass_accepts_wildcard_scoped_selection(source: str) -> None:
    resolve_module(parse_program(source))


@pytest.mark.parametrize(
    "source",
    (
        "import library using Point::distance",
        "export library hiding Point::internal",
    ),
)
def test_scope_pass_accepts_path_selection_atoms(source: str) -> None:
    resolve_module(parse_program(source))


def test_scoped_declarations_do_not_generate_runtime_initializers() -> None:
    result = PipelineDriver().run("def A::f() -> int = 0\n()", default_stdlib=False)

    assert result.ok


def test_library_scope_regions_apply_entry_only_declaration_restrictions(tmp_path: Path) -> None:
    root = tmp_path / "modules"
    root.mkdir()
    write_module_file(root, "library", "scope A\nagent bot\nend A")

    result = PipelineDriver().run(
        "import library\n()",
        roots=RootSet(roots=frozenset({root})),
        default_stdlib=False,
    )

    assert not result.ok


@pytest.mark.parametrize(
    ("entry", "library"),
    (
        ("import library using Point::distance\n()", "record Point()"),
        ("export library hiding Point::distance\n()", "record Point()"),
        ("import library\n()", "import dependency using Point::distance\ndef value() -> int = 0"),
    ),
)
def test_production_pipeline_validates_path_atoms_against_public_content(
    tmp_path: Path, entry: str, library: str
) -> None:
    root = tmp_path / "modules"
    root.mkdir()
    write_module_file(root, "library", library)
    if "dependency" in library:
        write_module_file(root, "dependency", "record Point()")

    result = PipelineDriver().run(
        entry,
        roots=RootSet(roots=frozenset({root})),
        default_stdlib=False,
    )

    assert not result.ok
    assert len(result.diagnostics) == 1
    assert "is not exported" in result.diagnostics[0].message
