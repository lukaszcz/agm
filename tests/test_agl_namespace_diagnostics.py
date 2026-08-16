"""Behavior coverage for namespace diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.modules.loader import ModuleGraph
from agm.agl.parser import parse_program
from agm.agl.parser.errors import AglSyntaxError
from agm.agl.scope.symbols import AglScopeError
from agm.agl.typecheck import AglTypeError
from agm.agl.typecheck.program import check_program
from tests.agl.ir_harness import base_caps, make_repl_graph_from_files, resolve_repl_graph


def _graph(tmp_path: Path, entry: str, modules: dict[str, str] | None = None) -> ModuleGraph:
    return make_repl_graph_from_files(tmp_path, {"entry": entry, **(modules or {})})


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


def test_bare_use_remains_an_expression() -> None:
    parse_program("use Tools")


def test_use_bare_target_ambiguity_suggests_a_reachable_module_anchor(tmp_path: Path) -> None:
    modules = {"library": "scope Scope\ndef remote() -> int = 1\nend Scope"}
    ambiguous = _graph(
        tmp_path,
        "import library::{Scope}\nuse Scope::*\nscope Scope\ndef local() -> int = 2\nend Scope",
        modules,
    )

    with pytest.raises(AglScopeError, match="ambiguous") as raised:
        resolve_repl_graph(ambiguous)

    diagnostic = str(raised.value)
    assert "/library::Scope" in diagnostic
    assert "::Scope" in diagnostic
    assert "/Scope" not in diagnostic

    module_route = _graph(
        tmp_path,
        "import library::{Scope}\nuse /library::Scope::*\nremote()",
        modules,
    )
    assert resolve_repl_graph(module_route).entry_id == module_route.entry_id

    local_scope = _graph(
        tmp_path,
        (
            "import library::{Scope}\nuse ::Scope::*\nscope Scope\ndef local() -> int = 2\n"
            "end Scope\nlocal()"
        ),
        modules,
    )
    assert resolve_repl_graph(local_scope).entry_id == local_scope.entry_id


def test_spaced_qualifier_near_miss_suggests_a_tight_qualifier(tmp_path: Path) -> None:
    graph = _graph(
        tmp_path,
        "import app/config\nconfig ::x",
        {"app/config": "def x() -> int = 1"},
    )

    with pytest.raises(AglScopeError) as raised:
        resolve_repl_graph(graph)

    diagnostic = str(raised.value).lower()
    assert "config::x" in diagnostic
    assert "whitespace" in diagnostic


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

    with pytest.raises(AglScopeError) as raised:
        resolve_repl_graph(graph)

    diagnostic = str(raised.value).lower()
    assert intended.lower() in diagnostic
    assert "whitespace" in diagnostic


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

    with pytest.raises(AglScopeError) as raised:
        resolve_repl_graph(graph)

    diagnostic = str(raised.value).lower()
    assert intended.lower() in diagnostic
    assert "whitespace" in diagnostic


def test_spaced_qualifier_repair_preserves_explicit_type_arguments(tmp_path: Path) -> None:
    """The suggested repair for a generic constructor remains well-typed."""
    module_source = "enum E[T]\n  | X"
    graph = _graph(
        tmp_path,
        "import app/config\napp/config ::E[int]::X",
        {"app/config": module_source},
    )

    with pytest.raises(AglScopeError) as raised:
        resolve_repl_graph(graph)

    assert "app/config::e[int]::x" in str(raised.value).lower()

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

    with pytest.raises(AglScopeError) as raised:
        resolve_repl_graph(graph)

    diagnostic = str(raised.value).lower()
    assert "app/config::x" in diagnostic
    assert "whitespace" in diagnostic


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
    "module_source",
    (
        "def other() -> int = 1\ndef x() -> int = 2",
        "def other() -> int = 1",
    ),
)
def test_spaced_qualifier_near_miss_requires_a_contributed_member(
    tmp_path: Path, module_source: str
) -> None:
    graph = _graph(
        tmp_path,
        "import app/config::other\nconfig ::x",
        {"app/config": module_source},
    )

    with pytest.raises(AglScopeError) as raised:
        resolve_repl_graph(graph)

    assert raised.value is not None


def test_spaced_type_qualified_near_miss_requires_a_constructible_owner(tmp_path: Path) -> None:
    graph = _graph(
        tmp_path,
        "import app/config\nconfig ::E::X",
        {"app/config": "type E = int"},
    )

    with pytest.raises(AglScopeError) as raised:
        resolve_repl_graph(graph)

    diagnostic = str(raised.value).lower()
    assert "whitespace" not in diagnostic
    assert "config::e::x" not in diagnostic


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

    with pytest.raises(AglScopeError) as raised:
        resolve_repl_graph(graph)

    diagnostic = str(raised.value).lower()
    assert intended.lower() in diagnostic
    assert "whitespace" in diagnostic


def test_an_unrelated_undefined_name_reports_its_own_error(tmp_path: Path) -> None:
    """A spaced qualifier elsewhere in the module does not colour other failures."""
    graph = _graph(
        tmp_path,
        "import app/config\nlet y = missing\nconfig ::x",
        {"app/config": "def x() -> int = 1"},
    )

    with pytest.raises(AglScopeError) as raised:
        resolve_repl_graph(graph)

    diagnostic = str(raised.value).lower()
    assert "missing" in diagnostic
    assert "whitespace" not in diagnostic


def test_spaced_qualifier_near_miss_reaches_a_non_juxtaposition_mis_parse(
    tmp_path: Path,
) -> None:
    """A spaced '::' behind a field access is not a bare call, but is still repaired."""
    graph = _graph(
        tmp_path,
        "import app/config\nrecord R\n  config: int\nlet r = R(config = 1)\nr.config ::x",
        {"app/config": "def x() -> int = 1"},
    )

    with pytest.raises(AglScopeError) as raised:
        resolve_repl_graph(graph)

    diagnostic = str(raised.value).lower()
    assert "config::x" in diagnostic
    assert "whitespace" in diagnostic


def test_qualified_scope_errors_distinguish_unknown_route_from_missing_member(
    tmp_path: Path,
) -> None:
    cases: tuple[tuple[str, dict[str, str], tuple[str, ...], tuple[str, ...]], ...] = (
        (
            "missing::read()",
            {},
            ("qualifier", "missing"),
            ("imported set",),
        ),
        (
            "import remote/config::read\nremote/config::missing()",
            {"remote/config": "def read() -> int = 1\nenum Flag | On"},
            ("remote/config", "not a public member", "is hidden", "missing"),
            ("qualifier",),
        ),
    )

    for entry, modules, expected, absent in cases:
        with pytest.raises(AglScopeError) as raised:
            resolve_repl_graph(_graph(tmp_path, entry, modules))
        diagnostic = str(raised.value).lower()
        assert all(term in diagnostic for term in expected)
        assert all(term not in diagnostic for term in absent)

    reachable = _graph(
        tmp_path,
        "import remote/config::read\nremote/config::Flag::On",
        {"remote/config": "def read() -> int = 1\nenum Flag | On"},
    )
    assert resolve_repl_graph(reachable).entry_id == reachable.entry_id


def test_qualified_type_errors_keep_unknown_route_and_missing_member_diagnostics(
    tmp_path: Path,
) -> None:
    cases: tuple[tuple[str, dict[str, str], tuple[str, ...]], ...] = (
        (
            "let value: missing::Item = null\nvalue",
            {},
            ("qualifier", "missing"),
        ),
        (
            "import remote/config::read\nlet value: remote/config::Missing = null\nvalue",
            {"remote/config": "def read() -> int = 1\nenum Flag | On"},
            ("remote/config", "accessible", "missing"),
        ),
    )

    for entry, modules, expected in cases:
        graph = _graph(tmp_path, entry, modules)
        with pytest.raises(AglTypeError) as raised:
            check_program(resolve_repl_graph(graph), base_caps())
        diagnostic = str(raised.value).lower()
        assert all(term in diagnostic for term in expected)

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
        "enum Flag\n"
        "  | Good\n"
        "end X\n"
        "\n"
        "scope Y\n"
        "enum Flag\n"
        "  | Good\n"
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

        with pytest.raises(AglTypeError) as raised:
            check_program(resolve_repl_graph(graph), base_caps())

        diagnostic = str(raised.value)
        assert "\x00" not in diagnostic
        assert "<entry>" not in diagnostic
        assert "X::Flag" in diagnostic
        assert "Y::Flag" in diagnostic
