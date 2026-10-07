"""Behavior coverage for namespace diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.modules.loader import ModuleGraph
from agm.agl.parser import parse_program
from agm.agl.parser.errors import AglSyntaxError
from agm.agl.scope.symbols import (
    AglScopeError,
    AmbiguousQualificationError,
    SpacedQualifierError,
    UnknownMemberError,
    UnknownQualifierError,
)
from agm.agl.typecheck.program import check_program
from tests.agl.ir_harness import base_caps, make_repl_graph_from_files, resolve_repl_graph


def _graph(tmp_path: Path, entry: str, modules: dict[str, str] | None = None) -> ModuleGraph:
    return make_repl_graph_from_files(tmp_path, {"entry": entry, **(modules or {})})


def _span_text(entry: str, error: AglScopeError) -> str:
    """The *entry* text *error*'s span covers."""
    assert error.span is not None
    return entry[error.span.start_offset : error.span.end_offset]


@pytest.mark.parametrize(
    "source",
    (
        "import old.path",
        "import /old/path",
        "import old/path qualified",
        "export /old/path",
        "export old/path qualified",
    ),
)
def test_legacy_module_header_spellings_are_syntax_errors(source: str) -> None:
    with pytest.raises(AglSyntaxError):
        parse_program(source)


def test_use_target_without_tail_or_alias_is_a_syntax_error() -> None:
    with pytest.raises(AglSyntaxError):
        parse_program("use Tools")


def test_anchored_use_targets_pick_the_module_route_or_the_own_scope(tmp_path: Path) -> None:
    modules = {"library": "scope Scope\n  def remote() -> int = 1\nend Scope"}
    module_route = _graph(
        tmp_path,
        "import library::{Scope}\nuse /library::Scope::*\nremote()",
        modules,
    )
    assert resolve_repl_graph(module_route).entry_id == module_route.entry_id

    local_scope = _graph(
        tmp_path,
        (
            "import library::{Scope}\nuse ::Scope::*\n\nscope Scope\n  def local() -> int = 2\n"
            "end Scope\n\nlocal()"
        ),
        modules,
    )
    assert resolve_repl_graph(local_scope).entry_id == local_scope.entry_id


@pytest.mark.parametrize(
    ("use", "spelling"),
    (
        ("use ::X::{Nope}", "::X::Nope"),
        ("use ::X::* hiding Nope", "::X::Nope"),
        ("use X::{Nope}", "X::Nope"),
        ("use a/lib::{Nope}", "a/lib::Nope"),
        ("use /a/lib::* hiding Nope", "/a/lib::Nope"),
    ),
)
def test_a_use_tail_miss_spells_the_target_as_written(
    tmp_path: Path, use: str, spelling: str
) -> None:
    graph = _graph(
        tmp_path,
        f"import a/lib\n{use}\n\nscope X\n  def f() -> int = 1\nend X\n\n()",
        {"a/lib": "def f() -> int = 1"},
    )
    with pytest.raises(UnknownMemberError) as raised:
        resolve_repl_graph(graph)
    assert type(raised.value) is UnknownMemberError
    assert raised.value.spelling == spelling


def test_spaced_qualifier_near_miss_suggests_a_tight_qualifier(tmp_path: Path) -> None:
    graph = _graph(
        tmp_path,
        "import app/config\nconfig ::x",
        {"app/config": "def x() -> int = 1"},
    )

    with pytest.raises(SpacedQualifierError) as raised:
        resolve_repl_graph(graph)

    assert raised.value.repair == "config::x"


def test_spaced_qualifier_near_miss_repairs_an_own_scope_member(tmp_path: Path) -> None:
    """A tight spelling selecting an own scope's member is repaired like a module route."""
    entry = "scope lib\n  def x() -> int = 1\nend lib\n\nlib ::x()"

    with pytest.raises(SpacedQualifierError) as raised:
        resolve_repl_graph(_graph(tmp_path, entry))

    assert raised.value.repair == "lib::x"
    assert _span_text(entry, raised.value) == "::"


def test_spaced_qualifier_near_miss_repairs_a_bare_imported_type_member(tmp_path: Path) -> None:
    """A spelling an import tail makes available bare is repaired like a module route."""
    entry = "import lib::*\nGeo ::Point(x = 1)"
    graph = _graph(tmp_path, entry, {"lib": "record Geo\n  y: int\nrecord Geo::Point\n  x: int"})

    with pytest.raises(SpacedQualifierError) as raised:
        resolve_repl_graph(graph)

    assert raised.value.repair == "Geo::Point"
    assert _span_text(entry, raised.value) == "::"


@pytest.mark.parametrize(
    ("spaced", "module_source", "intended"),
    (
        ("config ::x().field", "def x() -> int = 1", "config::x"),
        ("config ::xs()[0]", "def xs() -> array[int] = [1]", "config::xs"),
        ("config ::E::X", "enum E\n  | X", "config::E::X"),
    ),
)
def test_spaced_qualifier_near_miss_unwraps_postfix_and_type_qualifiers(
    tmp_path: Path, spaced: str, module_source: str, intended: str
) -> None:
    graph = _graph(
        tmp_path,
        f"import app/config\n{spaced}",
        {"app/config": module_source},
    )

    with pytest.raises(SpacedQualifierError) as raised:
        resolve_repl_graph(graph)

    assert raised.value.repair == intended


@pytest.mark.parametrize(
    ("spaced", "module_source", "intended"),
    (
        ("app/config ::x().field", "def x() -> int = 1", "app/config::x"),
        ("app/config ::E::X", "enum E\n  | X", "app/config::E::X"),
        ("app/config/tools ::x", "def x() -> int = 1", "app/config/tools::x"),
    ),
)
def test_spaced_slash_qualifier_near_miss_suggests_the_full_tight_route(
    tmp_path: Path, spaced: str, module_source: str, intended: str
) -> None:
    graph = _graph(
        tmp_path,
        f"import app/config\nimport app/config/tools\n{spaced}",
        {"app/config": module_source, "app/config/tools": module_source},
    )

    with pytest.raises(SpacedQualifierError) as raised:
        resolve_repl_graph(graph)

    assert raised.value.repair == intended


def test_spaced_qualifier_repair_preserves_explicit_type_arguments(tmp_path: Path) -> None:
    """The suggested repair for a generic constructor remains well-typed."""
    module_source = "enum E[T]\n  | X"
    graph = _graph(
        tmp_path,
        "import app/config\napp/config ::E[int]::X",
        {"app/config": module_source},
    )

    with pytest.raises(SpacedQualifierError) as raised:
        resolve_repl_graph(graph)

    assert raised.value.repair == "app/config::E[int]::X"

    repaired = resolve_repl_graph(
        _graph(
            tmp_path,
            "import app/config\napp/config::E[int]::X",
            {"app/config": module_source},
        )
    )
    check_program(repaired, base_caps())


def test_spaced_slash_qualifier_near_miss_uses_the_full_route_despite_suffix_collision(
    tmp_path: Path,
) -> None:
    graph = _graph(
        tmp_path,
        "import app/config\nimport other/config\napp/config ::x",
        {
            "app/config": "def x() -> int = 1",
            "other/config": "def x() -> int = 2",
        },
    )

    with pytest.raises(SpacedQualifierError) as raised:
        resolve_repl_graph(graph)

    assert raised.value.repair == "app/config::x"


@pytest.mark.parametrize(
    ("entry", "modules"),
    (
        (
            "import app/config\nlet config = fn(value: int) -> int => value\nlet x = 1\nconfig ::x",
            {"app/config": "def x() -> int = 1"},
        ),
        (
            (
                "import app/config\nlet app = 8\nlet x = 1\n"
                "def config(value: int) -> int = value\napp/config ::x"
            ),
            {"app/config": "def x() -> int = 1"},
        ),
        (
            "import app/config\nimport app/commands::config\nlet x = 1\nconfig ::x",
            {
                "app/config": "def x() -> int = 1",
                "app/commands": "def config(value: int) -> int = value",
            },
        ),
        (
            "import app/config\nimport app/commands::*\nlet x = 1\nconfig ::x",
            {
                "app/config": "def x() -> int = 1",
                "app/commands": "def config(value: int) -> int = value",
            },
        ),
        (
            "import app/print\nlet x = 1\nprint ::x",
            {"app/print": "def x() -> int = 1"},
        ),
        (
            (
                "import app/config\nrecord R\n  field: int\n"
                "def x() -> R = R(field = 1)\n"
                "def config(value: int) -> int = value\nconfig ::x().field"
            ),
            {"app/config": "def x() -> int = 1"},
        ),
        (
            (
                "import app/config\ndef xs() -> array[int] = [1]\n"
                "def config(value: int) -> int = value\nconfig ::xs()[0]"
            ),
            {"app/config": "def xs() -> array[int] = [1]"},
        ),
        (
            "import app/config\nenum E\n  | X\ndef config(value: E) -> E = value\nconfig ::E::X",
            {"app/config": "enum E\n  | X"},
        ),
    ),
)
def test_spaced_qualifier_preserves_resolvable_juxtaposition(
    tmp_path: Path, entry: str, modules: dict[str, str]
) -> None:
    resolved = resolve_repl_graph(_graph(tmp_path, entry, modules))
    check_program(resolved, base_caps())


@pytest.mark.parametrize(
    "expression",
    (
        "app / config(x)",
        "app / config ::x",
        "app / base/config ::x",
        "(app + base)/config ::x",
        "app / (base + extra)/config ::x",
        "::app/config ::x",
    ),
)
def test_spaced_slash_qualifier_preserves_non_route_division_forms(
    tmp_path: Path, expression: str
) -> None:
    resolved = resolve_repl_graph(
        _graph(
            tmp_path,
            (
                "import app/config\nlet app = 8\nlet base = 2\nlet extra = 2\nlet x = 1\n"
                f"def config(value: int) -> int = value\n{expression}"
            ),
            {"app/config": "def x() -> int = 1"},
        )
    )
    check_program(resolved, base_caps())


@pytest.mark.parametrize(
    ("module_source", "repair"),
    (
        ("def other() -> int = 1\ndef x() -> int = 2", "config::x"),
        ("def other() -> int = 1", None),
    ),
)
def test_spaced_qualifier_near_miss_requires_a_contributed_member(
    tmp_path: Path, module_source: str, repair: str | None
) -> None:
    entry = "import app/config::other\nconfig ::x"

    with pytest.raises(AglScopeError) as raised:
        resolve_repl_graph(_graph(tmp_path, entry, {"app/config": module_source}))

    # The tight-qualifier repair is offered only when the route really
    # contributes the member; otherwise the qualifier is just undefined.
    error = raised.value
    assert (error.repair if isinstance(error, SpacedQualifierError) else None) == repair
    assert _span_text(entry, error) == ("::" if repair else "config")


def test_spaced_type_qualified_near_miss_requires_a_constructible_owner(tmp_path: Path) -> None:
    entry = "import app/config\nconfig ::E::X"

    with pytest.raises(AglScopeError) as raised:
        resolve_repl_graph(_graph(tmp_path, entry, {"app/config": "type E = int"}))

    assert type(raised.value) is AglScopeError
    assert _span_text(entry, raised.value) == "config"


@pytest.mark.parametrize(
    ("entry", "modules", "intended"),
    (
        # The spelling before the spaced '::' is also a built-in call name.
        (
            "import app/print\nprint ::x",
            {"app/print": "def x() -> int = 1"},
            "print::x",
        ),
        # The spelling before the spaced '::' is also a local function.
        (
            "import app/config\ndef config(value: int) -> int = value\nconfig ::x",
            {"app/config": "def x() -> int = 1"},
            "config::x",
        ),
        # Every segment of the spaced route is also a local binding.
        (
            (
                "import app/config\nlet app = 8\n"
                "def config(value: int) -> int = value\napp/config ::x"
            ),
            {"app/config": "def x() -> int = 1"},
            "app/config::x",
        ),
        # A type-qualified member behind a locally-shadowed route spelling.
        (
            "import app/config\ndef config(value: int) -> int = value\nconfig ::E::X",
            {"app/config": "enum E\n  | X"},
            "config::E::X",
        ),
    ),
)
def test_spaced_qualifier_near_miss_survives_a_shadowed_route_spelling(
    tmp_path: Path, entry: str, modules: dict[str, str], intended: str
) -> None:
    """The repair is offered even when the spaced spelling has its own local meaning."""
    graph = _graph(tmp_path, entry, modules)

    with pytest.raises(SpacedQualifierError) as raised:
        resolve_repl_graph(graph)

    assert raised.value.repair == intended


def test_an_unrelated_undefined_name_reports_its_own_error(tmp_path: Path) -> None:
    """A spaced qualifier elsewhere in the module does not colour other failures."""
    entry = "import app/config\nlet y = missing\nconfig ::x"

    with pytest.raises(AglScopeError) as raised:
        resolve_repl_graph(_graph(tmp_path, entry, {"app/config": "def x() -> int = 1"}))

    assert type(raised.value) is AglScopeError
    assert _span_text(entry, raised.value) == "missing"


def test_spaced_qualifier_near_miss_reaches_a_non_juxtaposition_mis_parse(
    tmp_path: Path,
) -> None:
    """A spaced '::' behind a field access is not a bare call, but is still repaired."""
    graph = _graph(
        tmp_path,
        "import app/config\nrecord R\n  config: int\nlet r = R(config = 1)\nr.config ::x",
        {"app/config": "def x() -> int = 1"},
    )

    with pytest.raises(SpacedQualifierError) as raised:
        resolve_repl_graph(graph)

    assert raised.value.repair == "config::x"


def test_qualified_scope_errors_distinguish_unknown_route_from_missing_member(
    tmp_path: Path,
) -> None:
    with pytest.raises(UnknownQualifierError) as route:
        resolve_repl_graph(_graph(tmp_path, "missing::read()", {}))
    assert route.value.qualifier == "missing"

    with pytest.raises(UnknownMemberError) as member:
        resolve_repl_graph(
            _graph(
                tmp_path,
                "import remote/config::read\nremote/config::missing()",
                {"remote/config": "def read() -> int = 1\nenum Flag | On"},
            )
        )
    assert member.value.spelling == "remote/config::missing"

    reachable = _graph(
        tmp_path,
        "import remote/config::read\nremote/config::Flag::On",
        {"remote/config": "def read() -> int = 1\nenum Flag | On"},
    )
    assert resolve_repl_graph(reachable).entry_id == reachable.entry_id


def test_qualified_type_errors_keep_unknown_route_and_missing_member_diagnostics(
    tmp_path: Path,
) -> None:
    """An unknown route, or a route missing the named member, is a scope error."""
    cases: tuple[tuple[str, dict[str, str], type[AglScopeError]], ...] = (
        ("let value: missing::Item = null\nvalue", {}, UnknownQualifierError),
        (
            "import remote/config::read\nlet value: remote/config::Missing = null\nvalue",
            {"remote/config": "def read() -> int = 1\nenum Flag | On"},
            UnknownMemberError,
        ),
    )

    for entry, modules, error_type in cases:
        graph = _graph(tmp_path, entry, modules)
        with pytest.raises(error_type):
            resolve_repl_graph(graph)

    reachable = _graph(
        tmp_path,
        (
            "import remote/config::read\n"
            "let value: remote/config::Flag = remote/config::Flag::On\n"
            "value"
        ),
        {"remote/config": "def read() -> int = 1\nenum Flag | On"},
    )
    assert check_program(resolve_repl_graph(reachable), base_caps()).entry_id == reachable.entry_id


def test_qualified_ambiguity_lists_sorted_candidates_and_repairs(tmp_path: Path) -> None:
    graph = _graph(
        tmp_path,
        "import z/config\nimport a/config\nconfig::read()",
        {
            "a/config": "def read() -> int = 1",
            "z/config": "def read() -> int = 2",
        },
    )

    with pytest.raises(AglScopeError) as raised:
        resolve_repl_graph(graph)

    diagnostic = str(raised.value)
    assert diagnostic.index("a/config") < diagnostic.index("z/config")
    assert all(term in diagnostic for term in ("hiding", "longer suffix", "/-anchored", "as"))


def test_wildcard_tail_identifies_the_module_missing_the_selected_name(tmp_path: Path) -> None:
    graph = _graph(
        tmp_path,
        "import plugins/*::shared\nshared()",
        {
            "plugins/alpha": "def shared() -> int = 1",
            "plugins/beta": "def other() -> int = 2",
        },
    )

    with pytest.raises(AglScopeError) as raised:
        resolve_repl_graph(graph)

    diagnostic = str(raised.value)
    assert "shared" in diagnostic
    assert "plugins/beta" in diagnostic


class TestAmbiguityRepairsAreSpellable:
    """An ambiguity diagnostic suggests spellings the reader can actually type.

    The entry module's id carries a reserved segment holding a NUL byte so it
    can never collide with a real file, and it has no name a program can write.
    Neither may appear in a suggested repair: a declaration in the reading
    module is spelled by its scope path alone.
    """

    _AMBIGUOUS_SCOPE_USES = (
        "use X::*\n"
        "use Y::*\n"
        "\n"
        "scope X\n"
        "  enum Flag\n"
        "    | Good\n"
        "end X\n"
        "\n"
        "scope Y\n"
        "  enum Flag\n"
        "    | Good\n"
        "end Y\n"
    )

    def test_ambiguous_constructor_across_used_scopes(self, tmp_path: Path) -> None:
        graph = _graph(tmp_path, self._AMBIGUOUS_SCOPE_USES + "\nlet f = Flag::Good\n")

        with pytest.raises(AglScopeError) as raised:
            resolve_repl_graph(graph)

        diagnostic = str(raised.value)
        assert "\x00" not in diagnostic
        assert "<entry>" not in diagnostic
        assert "X::Flag::Good" in diagnostic
        assert "Y::Flag::Good" in diagnostic

    def test_ambiguous_type_across_used_scopes(self, tmp_path: Path) -> None:
        graph = _graph(
            tmp_path,
            self._AMBIGUOUS_SCOPE_USES + "\nlet f: X::Flag = X::Flag::Good\nlet g: Flag = f\n",
        )

        with pytest.raises(AmbiguousQualificationError) as raised:
            resolve_repl_graph(graph)

        diagnostic = str(raised.value)
        assert "\x00" not in diagnostic
        assert "<entry>" not in diagnostic
        assert "X::Flag" in diagnostic
        assert "Y::Flag" in diagnostic
