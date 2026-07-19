"""Behavior coverage for namespace diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.modules.loader import ModuleGraph
from agm.agl.parser import parse_program
from agm.agl.parser.errors import AglSyntaxError
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import AglScopeError
from agm.agl.typecheck import AglTypeError
from agm.agl.typecheck.program import check_program
from tests.agl.ir_harness import base_caps, make_graph_from_files


def _graph(tmp_path: Path, entry: str, modules: dict[str, str] | None = None) -> ModuleGraph:
    return make_graph_from_files(tmp_path, {"entry": entry, **(modules or {})})


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


def test_open_import_using_reports_the_redundant_combination() -> None:
    with pytest.raises(AglSyntaxError) as raised:
        parse_program("open import tools/text using trim")

    diagnostic = str(raised.value).lower()
    assert "open" in diagnostic
    assert "using" in diagnostic


def test_spaced_qualifier_near_miss_suggests_a_tight_qualifier(tmp_path: Path) -> None:
    graph = _graph(
        tmp_path,
        "import app/config\nconfig ::x",
        {"app/config": "def x() -> int = 1"},
    )

    with pytest.raises(AglScopeError) as raised:
        resolve_program(graph)

    diagnostic = str(raised.value).lower()
    assert "config::x" in diagnostic
    assert "whitespace" in diagnostic


@pytest.mark.parametrize(
    ("spaced", "module_source", "intended"),
    (
        ("config ::x().field", "def x() -> int = 1", "config::x"),
        ("config ::xs()[0]", "def xs() -> list[int] = [1]", "config::xs"),
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
        resolve_program(graph)

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
        resolve_program(graph)

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
        resolve_program(graph)

    assert "app/config::e[int]::x" in str(raised.value).lower()

    repaired = resolve_program(
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
        resolve_program(graph)

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
            "import app/config\nimport app/commands using config\nlet x = 1\nconfig ::x",
            {
                "app/config": "def x() -> int = 1",
                "app/commands": "def config(value: int) -> int = value",
            },
        ),
        (
            "import app/config\nopen import app/commands\nlet x = 1\nconfig ::x",
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
                "import app/config\ndef xs() -> list[int] = [1]\n"
                "def config(value: int) -> int = value\nconfig ::xs()[0]"
            ),
            {"app/config": "def xs() -> list[int] = [1]"},
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
    resolved = resolve_program(_graph(tmp_path, entry, modules))
    check_program(resolved, base_caps())


@pytest.mark.parametrize(
    "expression",
    (
        "app / config(x)",
        "app / config ::x",
        "app / base/config ::x",
        "(app + base)/config ::x",
        "app/(base + extra)/config ::x",
        "::app/config ::x",
    ),
)
def test_spaced_slash_qualifier_preserves_non_route_division_forms(
    tmp_path: Path, expression: str
) -> None:
    resolved = resolve_program(
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
        "import app/config using other\nconfig ::x",
        {"app/config": module_source},
    )

    with pytest.raises(AglScopeError) as raised:
        resolve_program(graph)

    diagnostic = str(raised.value).lower()
    assert "whitespace" not in diagnostic
    assert "config::x" not in diagnostic


def test_spaced_type_qualified_near_miss_requires_a_constructible_owner(tmp_path: Path) -> None:
    graph = _graph(
        tmp_path,
        "import app/config\nconfig ::E::X",
        {"app/config": "type E = int"},
    )

    with pytest.raises(AglScopeError) as raised:
        resolve_program(graph)

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
        resolve_program(graph)

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
        resolve_program(graph)

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
        resolve_program(graph)

    diagnostic = str(raised.value).lower()
    assert "config::x" in diagnostic
    assert "whitespace" in diagnostic


def test_qualified_scope_errors_distinguish_route_set_and_private_access(tmp_path: Path) -> None:
    cases: tuple[tuple[str, dict[str, str], tuple[str, ...], tuple[str, ...]], ...] = (
        (
            "missing::read()",
            {},
            ("qualifier", "missing"),
            ("private", "imported set"),
        ),
        (
            "import app/config using read\nconfig::write()",
            {"app/config": "def read() -> int = 1\ndef write() -> int = 2"},
            ("config", "imported set", "write"),
            ("private",),
        ),
        (
            "import app/config\nconfig::secret()",
            {"app/config": "def read() -> int = 1\nprivate def secret() -> int = 2"},
            ("config", "private", "secret"),
            ("imported set",),
        ),
    )

    for entry, modules, expected, absent in cases:
        with pytest.raises(AglScopeError) as raised:
            resolve_program(_graph(tmp_path, entry, modules))
        diagnostic = str(raised.value).lower()
        assert all(term in diagnostic for term in expected)
        assert all(term not in diagnostic for term in absent)


def test_qualified_type_errors_keep_the_same_distinctions(tmp_path: Path) -> None:
    cases: tuple[tuple[str, dict[str, str], tuple[str, ...]], ...] = (
        (
            "let value: missing::Item = null\nvalue",
            {},
            ("qualifier", "missing"),
        ),
        (
            "import app/types using Public\nlet value: types::Hidden = null\nvalue",
            {"app/types": "record Public\n  value: int\nrecord Hidden\n  value: int"},
            ("types", "accessible", "hidden"),
        ),
        (
            "import app/types\nlet value: types::Secret = null\nvalue",
            {"app/types": "record Public\n  value: int\nprivate record Secret\n  value: int"},
            ("types", "private", "secret"),
        ),
    )

    for entry, modules, expected in cases:
        graph = _graph(tmp_path, entry, modules)
        with pytest.raises(AglTypeError) as raised:
            check_program(resolve_program(graph), base_caps())
        diagnostic = str(raised.value).lower()
        assert all(term in diagnostic for term in expected)


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
        resolve_program(graph)

    diagnostic = str(raised.value)
    assert diagnostic.index("a/config") < diagnostic.index("z/config")
    assert all(term in diagnostic for term in ("hiding", "longer suffix", "/-anchored", "as"))


def test_wildcard_using_identifies_the_module_missing_the_selected_name(tmp_path: Path) -> None:
    graph = _graph(
        tmp_path,
        "import plugins/* using shared\nshared()",
        {
            "plugins/alpha": "def shared() -> int = 1",
            "plugins/beta": "def other() -> int = 2",
        },
    )

    with pytest.raises(AglScopeError) as raised:
        resolve_program(graph)

    diagnostic = str(raised.value)
    assert "shared" in diagnostic
    assert "plugins/beta" in diagnostic
