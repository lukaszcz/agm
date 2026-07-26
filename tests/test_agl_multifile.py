"""End-to-end tests for AgL multi-file (module-graph) programs.

Tests the program pipeline: ``prepare_program`` → ``run_prepared``, using
multi-module programs that exercise wildcard imports, qualified imports, library
roots, agent values passed to imported functions, and error diagnostics from
imported modules.

Each test uses ``tmp_path`` for concurrency safety.  Static test programs live
in ``tests/agl/multi_file/`` (outside ``programs/`` to avoid the module
e2e harness picking them up).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.program import resolve_program
from tests.agl.ir_harness import make_graph_from_files as _make_graph_from_files

MULTI_FILE_DIR = Path(__file__).parent / "agl" / "multi_file"
REPO_STDLIB_ROOT = Path(__file__).resolve().parents[1] / "stdlib"


def _make_runtime(
    *,
    default_agent: Any | None = None,
) -> Any:
    """Build a PipelineDriver with optional default agent."""
    from agm.agl import PipelineDriver

    return PipelineDriver(
        default_agent=default_agent,
    )


def _run_program(
    entry_source: str,
    *,
    roots_dirs: list[Path],
    entry_path: Path | None = None,
    param_values: dict[str, object] | None = None,
    default_agent: Any | None = None,
    agents: dict[str, Any] | None = None,
) -> Any:
    """Run a multi-file AgL program and return the RunResult."""
    from agm.agl import PipelineDriver
    from agm.agl.modules.roots import RootSet

    roots = RootSet(
        roots=frozenset({*(d.resolve() for d in roots_dirs if d.exists()), REPO_STDLIB_ROOT})
    )
    prepared = PipelineDriver.prepare_program(entry_source, entry_path=entry_path, roots=roots)
    rt = _make_runtime(default_agent=default_agent)
    if agents:
        for name, fn in agents.items():
            declarations = [item for item in prepared.declared_agents if item.name == name]
            if len(declarations) == 1:
                rt.register_scoped_agent(declarations[0].scope_path, name, fn)
            else:
                rt.register_agent(name, fn)
    return rt.run_prepared(prepared, param_values=param_values)


# ---------------------------------------------------------------------------
# Scenario 1: wildcard import (import utils/*)
# ---------------------------------------------------------------------------


def test_specific_catch_uses_module_qualified_exception_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A same-named local catch must not catch a distinct imported exception."""
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "lib.agl").write_text("exception Boom extends Exception\n  detail: text\n")

    source = (
        "import lib\n"
        "exception Boom extends Exception\n"
        "  code: int\n"
        "try\n"
        '  raise lib::Boom(message = "lib", detail = "from lib")\n'
        "catch Boom =>\n"
        '  print "wrong"\n'
        "catch _ =>\n"
        '  print "ok"\n'
    )
    result = _run_program(source, roots_dirs=[lib_dir])

    assert result.ok is True
    captured = capsys.readouterr()
    assert captured.out == "ok\n"


class TestWildcardImport:
    """import utils.* brings all modules from utils/ into scope."""

    def test_wildcard_import_basic_success(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Wildcard import: functions from all matched modules are accessible."""
        lib_dir = tmp_path / "lib"
        utils_dir = lib_dir / "utils"
        utils_dir.mkdir(parents=True)
        (utils_dir / "math.agl").write_text("def add(a: int, b: int) -> int = a + b\n")
        (utils_dir / "strings.agl").write_text(
            'def greet(name: text) -> text = "Hello, " + name + "!"\n'
        )

        source = (
            'open import utils/*\nlet n = add(3, 4)\nlet msg = greet("World")\nprint n\nprint msg\n'
        )
        result = _run_program(source, roots_dirs=[lib_dir])

        assert result.ok is True
        captured = capsys.readouterr()
        assert "7" in captured.out
        assert "Hello, World!" in captured.out

    def test_wildcard_import_from_repo_fixtures(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Wildcard import: uses multi_file fixture library in tests/agl/multi_file/."""
        entry_file = MULTI_FILE_DIR / "entry_wildcard.agl"
        source = entry_file.read_text()
        result = _run_program(source, entry_path=entry_file, roots_dirs=[MULTI_FILE_DIR])

        assert result.ok is True
        captured = capsys.readouterr()
        assert "7" in captured.out
        assert "Hello, World!" in captured.out


# ---------------------------------------------------------------------------
# Scenario 2: qualified import (import ... qualified)
# ---------------------------------------------------------------------------


class TestQualifiedImport:
    """import ... qualified requires :: qualifier to access names."""

    def test_qualified_import_via_full_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Qualified import: names accessible only with module::name syntax."""
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "calc.agl").write_text("def square(n: int) -> int = n * n\n")

        source = "open import calc\nlet r = calc::square(5)\nprint r\n"
        result = _run_program(source, roots_dirs=[lib_dir])

        assert result.ok is True
        captured = capsys.readouterr()
        assert "25" in captured.out

    def test_qualified_import_from_repo_fixtures(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Qualified import: uses multi_file fixture library."""
        entry_file = MULTI_FILE_DIR / "entry_qualified.agl"
        source = entry_file.read_text()
        result = _run_program(source, entry_path=entry_file, roots_dirs=[MULTI_FILE_DIR])

        assert result.ok is True
        captured = capsys.readouterr()
        assert "25" in captured.out

    def test_qualified_import_unqualified_access_fails(self, tmp_path: Path) -> None:
        """Unqualified access to a qualified-only import is a scope error."""
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "calc.agl").write_text("def square(n: int) -> int = n * n\n")

        source = "import calc\nlet r = square(5)\nr\n"
        result = _run_program(source, roots_dirs=[lib_dir])
        assert result.ok is False


# ---------------------------------------------------------------------------
# Generic inference across module boundaries
# ---------------------------------------------------------------------------


def test_imported_generic_inside_generic_module_body_freshens_per_entry_occurrence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fixtures keep an imported generic use inside app's generic declaration."""
    entry_file = MULTI_FILE_DIR / "entry_inference.agl"
    result = _run_program(
        entry_file.read_text(), entry_path=entry_file, roots_dirs=[MULTI_FILE_DIR]
    )

    assert result.ok is True
    assert capsys.readouterr().out == "1\ntext\n"


def test_unannotated_import_cycle_recursion_runs_from_fixture(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Imported mutually recursive candidates close before the entry executes."""
    entry_file = MULTI_FILE_DIR / "entry_recursive_inference.agl"
    result = _run_program(
        entry_file.read_text(), entry_path=entry_file, roots_dirs=[MULTI_FILE_DIR]
    )

    assert result.ok is True
    assert capsys.readouterr().out == "true\ntrue\n"


def test_unannotated_dependency_signature_is_available_to_noncyclic_importer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An importer consumes its dependency's inferred recursive signature."""
    entry_file = MULTI_FILE_DIR / "entry_noncyclic_recursive_inference.agl"
    result = _run_program(
        entry_file.read_text(), entry_path=entry_file, roots_dirs=[MULTI_FILE_DIR]
    )

    assert result.ok is True
    assert capsys.readouterr().out == "25\n"


# ---------------------------------------------------------------------------
# Scenario 3: error in imported module reports correct file path
# ---------------------------------------------------------------------------


class TestImportedModuleErrors:
    """Errors in imported modules carry that module's file path."""

    def test_type_error_in_imported_module_names_file(self, tmp_path: Path) -> None:
        """A type error in an imported module diagnostic shows that module's path."""
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        bad_mod = lib_dir / "broken.agl"
        # Type error: add(text, text) → int is wrong, but the function body
        # passes a text where int is expected
        bad_mod.write_text('def bad(n: int) -> int = "not an int"\n')

        source = "open import broken\nlet r = bad(1)\nprint r\n"
        result = _run_program(source, roots_dirs=[lib_dir])
        assert result.ok is False
        # At least one diagnostic should mention the broken.agl file
        assert any("broken.agl" in (d.source_label or "") for d in result.diagnostics), (
            f"No diagnostic mentions broken.agl; got: {result.diagnostics}"
        )

    def test_scope_error_in_imported_module_names_file(self, tmp_path: Path) -> None:
        """A scope error in an imported module diagnostic includes the file label."""
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        bad_mod = lib_dir / "badscope.agl"
        bad_mod.write_text("def f() -> int = undefined_name\n")

        source = "open import badscope\nlet r = f()\nr\n"
        result = _run_program(source, roots_dirs=[lib_dir])
        assert result.ok is False
        assert any("badscope.agl" in (d.source_label or "") for d in result.diagnostics), (
            f"No diagnostic mentions badscope.agl; got: {result.diagnostics}"
        )


# ---------------------------------------------------------------------------
# Scenario 4: lib-root module (separate lib root directory)
# ---------------------------------------------------------------------------


class TestLibRootModule:
    """Modules from a separate lib-root directory are found correctly."""

    def test_lib_root_module_found(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A module in a lib root (not the invocation root) is found and runs."""
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "shared.agl").write_text("def double(n: int) -> int = n * 2\n")

        work_dir = tmp_path / "work"
        work_dir.mkdir()
        entry = work_dir / "prog.agl"
        entry.write_text("open import shared\nlet r = double(21)\nprint r\n")

        # Use lib_dir as the lib-root, work_dir as the invocation root.
        source = entry.read_text()
        result = _run_program(source, entry_path=entry, roots_dirs=[work_dir, lib_dir])
        assert result.ok is True
        captured = capsys.readouterr()
        assert "42" in captured.out

    def test_module_not_found_fails(self, tmp_path: Path) -> None:
        """A missing module causes a ModuleNotFound diagnostic."""
        source = "open import missing_module\nlet x = 1\nx\n"
        result = _run_program(source, roots_dirs=[tmp_path])
        assert result.ok is False
        assert any("missing_module" in d.message for d in result.diagnostics)

    def test_ambiguous_module_fails(self, tmp_path: Path) -> None:
        """A module found in two roots is an AmbiguousModule error."""
        root_a = tmp_path / "root_a"
        root_b = tmp_path / "root_b"
        root_a.mkdir()
        root_b.mkdir()
        (root_a / "shared.agl").write_text("def f() -> int = 1\n")
        (root_b / "shared.agl").write_text("def f() -> int = 2\n")

        source = "open import shared\nlet r = f()\nr\n"
        result = _run_program(source, roots_dirs=[root_a, root_b])
        assert result.ok is False
        assert any("shared" in d.message for d in result.diagnostics)


# ---------------------------------------------------------------------------
# Scenario 5: agent value passed to imported function
# ---------------------------------------------------------------------------


class TestAgentValueCrossModule:
    """Agent values declared in entry can be passed to imported library functions."""

    def test_agent_passed_to_imported_function(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Agent declared in entry is passed as a value to an imported function."""
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "helper.agl").write_text(
            "def ask_with_agent(prompt: text, a: agent) -> text = ask(prompt, agent = a)\n"
        )

        source = (
            "agent mybot\n"
            "open import helper\n"
            'let result = ask_with_agent("test question", mybot)\n'
            "print result\n"
        )

        responses: list[str] = ["mocked answer"]

        def scripted_agent(req: Any) -> str:
            return responses[0]

        result = _run_program(
            source,
            roots_dirs=[lib_dir],
            default_agent=scripted_agent,
            agents={"mybot": scripted_agent},
        )
        assert result.ok is True
        captured = capsys.readouterr()
        assert "mocked answer" in captured.out

    def test_agent_value_in_entry_with_lib_module(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Agent from lib fixture: ask_with_agent works cross-module."""
        lib_dir = MULTI_FILE_DIR

        source = (
            "agent mybot\n"
            "open import utils/agent_helper\n"
            'let r = ask_with_agent("ping", mybot)\n'
            "print r\n"
        )

        def scripted_agent(req: Any) -> str:
            return "pong"

        result = _run_program(
            source,
            roots_dirs=[lib_dir],
            default_agent=scripted_agent,
            agents={"mybot": scripted_agent},
        )
        assert result.ok is True
        captured = capsys.readouterr()
        assert "pong" in captured.out


# ---------------------------------------------------------------------------
# Multi-module + params integration
# ---------------------------------------------------------------------------


class TestMultiFileParams:
    """params and imported modules work together."""

    def test_param_with_imported_function(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An entry with a param can call an imported function with that param."""
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "math.agl").write_text("def square(n: int) -> int = n * n\n")

        source = "open import math\nparam n: int\nlet r = square(n)\nprint r\n"

        from agm.agl import PipelineDriver
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset({lib_dir.resolve(), REPO_STDLIB_ROOT}))
        prepared = PipelineDriver.prepare_program(source, entry_path=None, roots=roots)

        rt = PipelineDriver()
        result = rt.run_prepared(prepared, param_values={"n": 7})
        assert result.ok is True
        captured = capsys.readouterr()
        assert "49" in captured.out

    def test_missing_param_in_multifile_fails(self, tmp_path: Path) -> None:
        """A missing required param in a multi-file program fails cleanly."""
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "calc.agl").write_text("def sq(n: int) -> int = n * n\n")

        source = "open import calc\nparam n: int\nlet r = sq(n)\nprint r\n"

        from agm.agl import PipelineDriver
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset({lib_dir.resolve(), REPO_STDLIB_ROOT}))
        prepared = PipelineDriver.prepare_program(source, entry_path=None, roots=roots)

        rt = PipelineDriver()
        result = rt.run_prepared(prepared, param_values={})
        assert result.ok is False
        assert any("n" in d.message for d in result.diagnostics)


# ---------------------------------------------------------------------------
# Scenario 7: wildcard import with using / hiding
# ---------------------------------------------------------------------------


class TestWildcardImportUsingHiding:
    """import pkg.* using … / hiding … works end-to-end through real source."""

    def _make_pkg(self, tmp_path: Path) -> Path:
        """Create a small package with two modules, each exporting two names."""
        lib_dir = tmp_path / "lib"
        pkg_dir = lib_dir / "pkg"
        pkg_dir.mkdir(parents=True)
        # pkg.math exports: add, mul
        (pkg_dir / "math.agl").write_text(
            "def add(a: int, b: int) -> int = a + b\ndef mul(a: int, b: int) -> int = a * b\n"
        )
        # pkg.text exports: upper (faked as concatenation), join
        (pkg_dir / "text.agl").write_text(
            'def join(a: text, b: text) -> text = a + " " + b\n'
            'def greet(name: text) -> text = "Hello, " + name\n'
        )
        return lib_dir

    def test_wildcard_using_restricts_names(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """import pkg.* using add brings only 'add' into scope; 'mul' is inaccessible."""
        lib_dir = self._make_pkg(tmp_path)
        # A wildcard `using` list must be public in every matched module.
        source = "import pkg/* using add\nlet r = add(3, 4)\nprint r\n"
        result = _run_program(source, roots_dirs=[lib_dir])
        assert result.ok is False
        assert result.diagnostics

    def test_wildcard_using_hidden_name_inaccessible(self, tmp_path: Path) -> None:
        """Names not listed in 'using' are inaccessible even though exported."""
        lib_dir = self._make_pkg(tmp_path)
        # Validation occurs at the wildcard import before body resolution.
        source = "import pkg/* using add\nlet r = mul(3, 4)\nprint r\n"
        result = _run_program(source, roots_dirs=[lib_dir])
        assert result.ok is False
        assert result.diagnostics

    def test_wildcard_hiding_removes_name(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """import pkg.* hiding mul brings all names except 'mul' into scope."""
        lib_dir = self._make_pkg(tmp_path)
        # A wildcard `hiding` list is likewise checked per expanded module.
        source = "open import pkg/* hiding mul\nlet r = add(10, 5)\nprint r\n"
        result = _run_program(source, roots_dirs=[lib_dir])
        assert result.ok is False
        assert result.diagnostics

    def test_wildcard_hiding_name_inaccessible(self, tmp_path: Path) -> None:
        """The hidden name is inaccessible after hiding."""
        lib_dir = self._make_pkg(tmp_path)
        source = "open import pkg/* hiding mul\nlet r = mul(3, 4)\nprint r\n"
        result = _run_program(source, roots_dirs=[lib_dir])
        assert result.ok is False
        assert any("mul" in d.message for d in result.diagnostics)

    def test_wildcard_using_multi_module_union(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """using selects a name from every matched module that exports it."""
        lib_dir = self._make_pkg(tmp_path)
        # A list that is not shared by every module is rejected at the import.
        source = (
            "import pkg/* using add, greet\n"
            "let n = add(2, 3)\n"
            'let s = greet("World")\n'
            "print n\n"
            "print s\n"
        )
        result = _run_program(source, roots_dirs=[lib_dir])
        assert result.ok is False
        assert result.diagnostics


# ---------------------------------------------------------------------------
# Scoped module selections
# ---------------------------------------------------------------------------


class TestScopedModuleSelections:
    """Scoped public members retain their paths across module boundaries."""

    def _write_geo(self, root: Path) -> None:
        source = MULTI_FILE_DIR / "scoped_selection" / "geo.agl"
        (root / "geo.agl").write_text(source.read_text())

    def test_selecting_a_scoped_member_keeps_its_full_route(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._write_geo(tmp_path)

        result = _run_program(
            "import geo using Point::distance\nlet value = geo::Point::distance()\nprint value\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is True
        assert capsys.readouterr().out == "7\n"

    def test_selecting_a_scope_exposes_its_public_subtree(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._write_geo(tmp_path)

        result = _run_program(
            "import geo using Point\nprint geo::Point::distance()\nprint geo::Point::bearing()\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is True
        assert capsys.readouterr().out == "7\n3\n"

    def test_variant_selection_filters_the_complete_path(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._write_geo(tmp_path)

        result = _run_program(
            "import geo using Point::Color::red\nprint geo::Point::Color::red\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is True
        assert capsys.readouterr().out == "Point::Color::red\n"

    def test_hiding_a_variant_removes_its_complete_path(self, tmp_path: Path) -> None:
        self._write_geo(tmp_path)

        result = _run_program(
            "import geo hiding Point::Color::red\ngeo::Point::Color::red\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is False

    def test_hiding_a_scope_removes_its_public_subtree(self, tmp_path: Path) -> None:
        self._write_geo(tmp_path)

        result = _run_program(
            "import geo hiding Point\ngeo::Point::distance()\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is False

    @pytest.mark.parametrize(
        ("selection", "reference"),
        (
            ("Point::distance as d", "d()"),
            ("Point::distance", "Point::distance()"),
            ("Point as P", "P::distance()"),
        ),
    )
    def test_scoped_using_contributes_full_paths(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        selection: str,
        reference: str,
    ) -> None:
        self._write_geo(tmp_path)

        result = _run_program(
            f"import geo using {selection}\nprint {reference}\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is True
        assert capsys.readouterr().out == "7\n"

    def test_scoped_generic_alias_resolves_before_its_origin_body(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "library.agl").write_text("type Point::Values[T] = list[T]\n")

        result = _run_program(
            "import library using Point\n"
            "record Holder\n"
            "  values: library::Point::Values[int]\n"
            "let holder = Holder(values = [7])\n"
            "print holder.values\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is True
        assert capsys.readouterr().out == "[7]\n"

    def test_rerooted_subtrees_with_different_origins_conflict_on_reexport(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "library.agl").write_text(
            "def Point::distance() -> int = 1\ndef Other::distance() -> int = 2\n"
        )
        (tmp_path / "facade.agl").write_text(
            "export library using Point as Location, Other as Location\n"
        )

        result = _run_program("import facade\n()\n", roots_dirs=[tmp_path])

        assert result.ok is False

    def test_rerooted_subtree_reexports_same_origin_through_multiple_facades(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "origin.agl").write_text("def Point::distance() -> int = 7\n")
        (tmp_path / "left.agl").write_text("export origin using Point\n")
        (tmp_path / "right.agl").write_text("export origin using Point\n")
        (tmp_path / "facade.agl").write_text(
            "export left using Point as Location\nexport right using Point as Location\n"
        )

        result = _run_program(
            "import facade using Location\nprint facade::Location::distance()\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is True
        assert capsys.readouterr().out == "7\n"

    def test_reexport_rename_preserves_scoped_origin_identity(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._write_geo(tmp_path)
        (tmp_path / "facade.agl").write_text("export geo using Point as Location\n")

        result = _run_program(
            "import facade using Location\nprint facade::Location::distance()\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is True
        assert capsys.readouterr().out == "7\n"

        graph = _make_graph_from_files(
            tmp_path,
            {
                "entry": "import facade using Location\n()",
                "facade": "export geo using Point as Location",
                "geo": (tmp_path / "geo.agl").read_text(),
            },
        )
        resolved = resolve_program(graph)
        assert resolved.modules[ModuleId.from_path("facade")].exports[("Location", "distance")] == (
            ModuleId.from_path("geo"),
            ("Point", "distance"),
        )

    def test_wildcard_import_distributes_scoped_selection(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._write_geo(tmp_path)

        result = _run_program(
            "import geo/* using Point::distance\nprint geo::Point::distance()\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is True
        assert capsys.readouterr().out == "7\n"

    def test_type_owned_members_and_nested_construction_execute(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._write_geo(tmp_path)

        result = _run_program(
            "import geo using Point, Shapes::Point\n"
            "let point = geo::Point(x = 4)\n"
            "let nested = geo::Shapes::Point(x = 5)\n"
            "print geo::Point::distance()\n"
            "print point.x\n"
            "print nested.x\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is True
        assert capsys.readouterr().out == "7\n4\n5\n"

    def test_renamed_scoped_record_constructs_via_its_bare_alias(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A scoped record selected and renamed by 'using … as' builds under its alias."""
        self._write_geo(tmp_path)

        result = _run_program(
            "import geo using Point as P\nlet p = P(x = 9)\nprint p.x\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is True
        assert capsys.readouterr().out == "9\n"

    def test_renamed_root_enum_variant_constructs_via_its_bare_alias(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A root enum variant individually selected and renamed builds as a value."""
        (tmp_path / "flags.agl").write_text("enum Status\n  | Good\n  | Bad\n")

        result = _run_program(
            "import flags using Status::Good as X\nlet s = X\nprint s\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is True
        assert capsys.readouterr().out == "Status::Good\n"

    def test_renamed_scoped_enum_exposes_its_variants_bare(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Renaming a scoped enum exposes its variants bare, as a root enum does."""
        self._write_geo(tmp_path)

        result = _run_program(
            "import geo using Point::Color as C\nprint C::red\nprint red\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is True
        assert capsys.readouterr().out == "Point::Color::red\nPoint::Color::red\n"

    def test_opening_a_bare_imported_scope_contributed_twice_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """Two bare-imported modules contributing the same scope leave `open` ambiguous."""
        (tmp_path / "lib1.agl").write_text("scope A\ndef one() -> int = 1\nend A\n")
        (tmp_path / "lib2.agl").write_text("scope A\ndef two() -> int = 2\nend A\n")

        result = _run_program(
            "open import lib1\nopen import lib2\nopen A\nprint one()\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is False
        assert result.diagnostics

    def test_scoped_generic_accepts_explicit_type_arguments_across_modules(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A module route onto a scoped generic is a route, not a type qualifier."""
        (tmp_path / "boxes.agl").write_text("scope A\nrecord Box[T](v: T)\nend A\n")

        result = _run_program(
            "import boxes\nlet build = boxes::A::Box::[int]\nprint build(3).v\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is True
        assert capsys.readouterr().out == "3\n"

    def test_selected_scoped_member_reaches_its_scope_siblings(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._write_geo(tmp_path)

        result = _run_program(
            "import geo using Point::public_secret\nprint geo::Point::public_secret()\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is True
        assert capsys.readouterr().out == "99\n"

    def test_selecting_a_scope_without_members_fails(self, tmp_path: Path) -> None:
        source = MULTI_FILE_DIR / "scoped_selection" / "empty.agl"
        (tmp_path / "geo.agl").write_text(source.read_text())

        result = _run_program(
            "import geo using Empty\n()\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is False


class TestCrossModuleScopedPaths:
    """Scoped path atoms retain module identity through imports."""

    def test_same_scoped_path_from_two_modules_clashes_only_at_use(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "north.agl").write_text("def Point::distance() -> int = 7\n")
        (tmp_path / "south.agl").write_text("def Point::distance() -> int = 9\n")

        result = _run_program(
            "open import north\n"
            "open import south\n"
            "print /north::Point::distance()\n"
            "print /south::Point::distance()\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is True
        assert capsys.readouterr().out == "7\n9\n"

        ambiguous = _run_program(
            "open import north\nopen import south\nPoint::distance()\n", roots_dirs=[tmp_path]
        )
        assert ambiguous.ok is False
        assert any(
            "ambiguous" in diagnostic.message.lower() for diagnostic in ambiguous.diagnostics
        )

    def test_module_route_and_open_scoped_path_clash_at_use(self, tmp_path: Path) -> None:
        (tmp_path / "Point.agl").write_text("def distance() -> int = 9\n")
        (tmp_path / "geometry.agl").write_text("def Point::distance() -> int = 7\n")

        ambiguous = _run_program(
            "import Point\nopen import geometry\nPoint::distance()\n", roots_dirs=[tmp_path]
        )
        assert ambiguous.ok is False
        assert any(
            "ambiguous" in diagnostic.message.lower() for diagnostic in ambiguous.diagnostics
        )

        repaired = _run_program(
            "import Point\nopen import geometry\nprint /Point::distance()\n",
            roots_dirs=[tmp_path],
        )
        assert repaired.ok is True

    def test_scope_and_module_route_clash_are_repaired_by_anchors(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "Point.agl").write_text("def distance() -> int = 9\n")

        ambiguous = _run_program(
            "import Point\nscope Point\ndef distance() -> int = 7\nend Point\nPoint::distance()\n",
            roots_dirs=[tmp_path],
        )
        assert ambiguous.ok is False
        assert any(
            "both" in diagnostic.message.lower() and "module route" in diagnostic.message.lower()
            for diagnostic in ambiguous.diagnostics
        )

        repaired = _run_program(
            "import Point\n"
            "scope Point\n"
            "def distance() -> int = 7\n"
            "end Point\n"
            "print /Point::distance()\n"
            "print ::Point::distance()\n",
            roots_dirs=[tmp_path],
        )
        assert repaired.ok is True
        assert capsys.readouterr().out == "9\n7\n"

    def test_open_import_exposes_scoped_paths_without_a_module_route(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "geo.agl").write_text(
            "def Point::distance() -> int = 7\n"
            "def Point::bearing() -> int = 3\n"
            "def Point::secret() -> int = 99\n"
            "def Point::public_secret() -> int = secret()\n"
        )

        result = _run_program(
            "open import geo hiding Point::bearing\n"
            "print Point::distance()\n"
            "print Point::public_secret()\n",
            roots_dirs=[tmp_path],
        )
        assert result.ok is True
        assert capsys.readouterr().out == "7\n99\n"

        full = _run_program("open import geo\nprint Point::bearing()\n", roots_dirs=[tmp_path])
        assert full.ok is True
        assert capsys.readouterr().out == "3\n"

    def test_wildcard_selection_distributes_scoped_paths_and_excludes_unselected_members(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        shapes = tmp_path / "shapes"
        shapes.mkdir()
        (shapes / "circle.agl").write_text(
            "def Point::measure() -> int = 3\ndef Point::secret() -> int = 30\n"
        )
        (shapes / "square.agl").write_text(
            "def Point::measure() -> int = 4\ndef Point::secret() -> int = 40\n"
        )

        result = _run_program(
            "import shapes/* using Point::measure\n"
            "print circle::Point::measure()\n"
            "print square::Point::measure()\n",
            roots_dirs=[tmp_path],
        )
        assert result.ok is True
        assert capsys.readouterr().out == "3\n4\n"

        unselected = _run_program(
            "import shapes/* using Point::measure\ncircle::Point::secret()\n", roots_dirs=[tmp_path]
        )
        assert unselected.ok is False


# ---------------------------------------------------------------------------
# extern def (Python FFI): a library module's extern reachable across
# qualified/open imports and re-export.
#
# Companion loading, boundary crossing, and the full conversion matrix are
# covered end to end elsewhere (test_agl_extern_runtime.py); this class only
# covers the multi-file import/export surface, using the repo fixture library
# at tests/agl/multi_file/utils/ext_math.agl (+ .py companion) and its
# re-export facade tests/agl/multi_file/utils/ext_facade.agl.
# ---------------------------------------------------------------------------


class TestOpenedScopes:
    """Opening local and imported named scopes exposes only selected members."""

    def test_imported_scope_opens_members_and_type_variants(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "geo.agl").write_text(
            "def Point::distance() -> int = 7\n"
            "def Point::secret() -> int = 99\n"
            "enum Flag | Ready\n"
            'def Flag::label() -> text = "ready"\n'
        )

        result = _run_program(
            "import geo\n"
            "open geo::Point\n"
            "open geo::Flag\n"
            "print distance()\n"
            "print Ready\n"
            "print label()\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is True
        assert capsys.readouterr().out == "7\nFlag::Ready\nready\n"

    def test_nearer_opened_enum_variant_keeps_its_scoped_owner(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = _run_program(
            "open A::Flag\n"
            "scope B\n"
            "open Flag\n"
            "def value() -> B::Flag = Ready\n"
            "enum Flag | Ready\n"
            "end B\n"
            "def root_value() -> A::Flag = Ready\n"
            "scope A\n"
            "enum Flag | Ready\n"
            "end A\n"
            "print B::value()\n"
            "print root_value()\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is True
        assert capsys.readouterr().out == "B::Flag::Ready\nA::Flag::Ready\n"

    def test_ambiguous_imported_scope_route_is_rejected(self, tmp_path: Path) -> None:
        for directory in ("one", "two"):
            module_dir = tmp_path / directory
            module_dir.mkdir()
            (module_dir / "geo.agl").write_text("def Point::distance() -> int = 1\n")

        result = _run_program(
            "import one/geo\nimport two/geo\nopen geo::Point\n()\n", roots_dirs=[tmp_path]
        )

        assert result.ok is False
        assert "ambiguous" in result.diagnostics[0].message

    def test_opened_variant_merges_with_the_same_selected_import(self, tmp_path: Path) -> None:
        (tmp_path / "geo.agl").write_text("enum Flag | Ready\n")

        result = _run_program(
            "import geo\nimport geo using Flag::Ready as Ready\nopen geo::Flag\nReady\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is True

    @pytest.mark.parametrize(
        "source",
        (
            "import geo\nopen geo::Point using missing\n()",
            "import geo\nopen geo::Missing\n()",
            "scope Point\ndef distance() -> int = 1\nend Point\nopen Point using missing\n()",
            "scope Point\ndef secret() -> int = 1\nend Point\nopen Point hiding secret\nsecret()",
        ),
    )
    def test_open_rejects_unknown_scopes_and_unselected_members(
        self, tmp_path: Path, source: str
    ) -> None:
        (tmp_path / "geo.agl").write_text(
            "def Point::distance() -> int = 7\ndef Point::secret() -> int = 99\n"
        )

        assert _run_program(source, roots_dirs=[tmp_path]).ok is False

    def test_scope_open_composes_with_open_import_without_transitive_reexport(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "geo.agl").write_text(
            "def root() -> int = 1\ndef Point::distance() -> int = 7\n"
        )
        (tmp_path / "facade.agl").write_text(
            "import geo\nopen geo::Point\ndef value() -> int = 2\n"
        )

        result = _run_program(
            "open import geo\nopen geo::Point\nprint root()\nprint distance()\n",
            roots_dirs=[tmp_path],
        )
        assert result.ok is True
        assert capsys.readouterr().out == "1\n7\n"

        assert _run_program("open import facade\ndistance()\n", roots_dirs=[tmp_path]).ok is False

    def test_open_scope_and_open_import_clash_when_the_name_is_used(self, tmp_path: Path) -> None:
        (tmp_path / "geo.agl").write_text(
            "def distance() -> int = 1\ndef Point::distance() -> int = 7\n"
        )

        result = _run_program(
            "open import geo\nopen geo::Point\ndistance()\n", roots_dirs=[tmp_path]
        )

        assert result.ok is False
        assert "ambiguous" in result.diagnostics[0].message

    def test_open_rename_collision_reports_each_contributing_member(self, tmp_path: Path) -> None:
        (tmp_path / "geo.agl").write_text(
            "def Point::distance() -> int = 1\ndef Point::length() -> int = 2\n"
        )

        result = _run_program(
            "import geo\nopen geo::Point using distance as measure, length as measure\nmeasure()\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is False
        assert "ambiguous" in result.diagnostics[0].message
        assert "Point::distance" in result.diagnostics[0].message
        assert "Point::length" in result.diagnostics[0].message

    def test_opened_local_scope_type_is_available_only_in_its_region(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = (
            "scope Shapes\n"
            "record Point\n"
            "  value: int\n"
            "end Shapes\n"
            "scope Measurements\n"
            "open Shapes\n"
            "def value(point: Point) -> int = point.value\n"
            "end Measurements\n"
            "print Measurements::value(Shapes::Point(value = 3))\n"
        )

        result = _run_program(source, roots_dirs=[tmp_path])

        assert result.ok is True
        assert capsys.readouterr().out == "3\n"

    def test_opened_scope_type_does_not_leak_from_its_region(self, tmp_path: Path) -> None:
        source = (
            "scope Shapes\n"
            "record Point\n"
            "  value: int\n"
            "end Shapes\n"
            "scope Measurements\n"
            "open Shapes\n"
            "def value(point: Point) -> int = point.value\n"
            "end Measurements\n"
            "def leaked(point: Point) -> int = point.value\n"
            "()\n"
        )

        assert _run_program(source, roots_dirs=[tmp_path]).ok is False

    def test_opened_imported_scope_type_honors_selection_and_rename(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "geo.agl").write_text("scope Shapes\nrecord Point\n  value: int\nend Shapes\n")

        result = _run_program(
            "import geo\n"
            "scope Measurements\n"
            "open geo::Shapes using Point as Coordinate\n"
            "def value(point: Coordinate) -> int = point.value\n"
            "end Measurements\n"
            "print Measurements::value(geo::Shapes::Point(value = 5))\n",
            roots_dirs=[tmp_path],
        )

        assert result.ok is True
        assert capsys.readouterr().out == "5\n"

    def test_opened_generic_type_is_resolved_by_its_bare_rename(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = (
            "open Shapes using Box as Container\n"
            "scope Shapes\n"
            "record Box[T]\n"
            "  value: T\n"
            "end Shapes\n"
            "def value(box: Container[int]) -> int = box.value\n"
            "print value(Shapes::Box(value = 7))\n"
        )

        result = _run_program(source, roots_dirs=[tmp_path])

        assert result.ok is True
        assert capsys.readouterr().out == "7\n"

    def test_opened_scope_function_does_not_resolve_as_a_type(self, tmp_path: Path) -> None:
        (tmp_path / "geo.agl").write_text("scope Shapes\ndef Point() -> int = 1\nend Shapes\n")

        source = "import geo\nopen geo::Shapes\ndef value(point: Point) -> int = point\n()\n"

        assert _run_program(source, roots_dirs=[tmp_path]).ok is False

    def test_opened_generic_type_requires_arguments(self, tmp_path: Path) -> None:
        source = (
            "open Shapes using Box as Container\n"
            "scope Shapes\n"
            "record Box[T]\n"
            "  value: T\n"
            "end Shapes\n"
            "def value(box: Container) -> int = box.value\n"
            "()\n"
        )

        assert _run_program(source, roots_dirs=[tmp_path]).ok is False

    def test_opened_non_generic_type_rejects_arguments(self, tmp_path: Path) -> None:
        source = (
            "open Shapes using Point as Coordinate\n"
            "scope Shapes\n"
            "record Point\n"
            "  value: int\n"
            "end Shapes\n"
            "def value(point: Coordinate[int]) -> int = point.value\n"
            "()\n"
        )

        assert _run_program(source, roots_dirs=[tmp_path]).ok is False

    def test_opened_type_rename_collision_is_ambiguous_at_its_use(self, tmp_path: Path) -> None:
        source = (
            "open First using Point as Coordinate\n"
            "open Second using Point as Coordinate\n"
            "scope First\nrecord Point\n  value: int\nend First\n"
            "scope Second\nrecord Point\n  value: int\nend Second\n"
            "def value(point: Coordinate) -> int = point.value\n"
            "()\n"
        )

        assert _run_program(source, roots_dirs=[tmp_path]).ok is False

    def test_opened_generic_alias_is_resolved_by_its_bare_rename(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = (
            "open Shapes using Wrapper as Container\n"
            "scope Shapes\n"
            "record Box[T]\n"
            "  value: T\n"
            "type Wrapper[T] = Shapes::Box[T]\n"
            "end Shapes\n"
            "def value(box: Container[int]) -> int = box.value\n"
            "print value(Shapes::Box(value = 9))\n"
        )

        result = _run_program(source, roots_dirs=[tmp_path])

        assert result.ok is True
        assert capsys.readouterr().out == "9\n"


class TestExternMultiFile:
    """A library module's ``extern def`` through qualified/open imports and re-export."""

    def test_qualified_import_calls_the_extern(self, capsys: pytest.CaptureFixture[str]) -> None:
        source = "open import utils/ext_math\nlet r = utils/ext_math::double(21)\nprint r\n"
        result = _run_program(source, roots_dirs=[MULTI_FILE_DIR])
        assert result.ok is True
        assert "42" in capsys.readouterr().out

    def test_open_import_calls_the_extern_unqualified(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = "open import utils/ext_math\nlet r = double(21)\nprint r\n"
        result = _run_program(source, roots_dirs=[MULTI_FILE_DIR])
        assert result.ok is True
        assert "42" in capsys.readouterr().out

    def test_extern_usable_from_an_agl_function_in_its_own_module(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # `use_secret` calls the extern `secret` internally, so the importer
        # reaches the extern through an ordinary AgL wrapper.
        source = "open import utils/ext_math\nlet r = use_secret(21)\nprint r\n"
        result = _run_program(source, roots_dirs=[MULTI_FILE_DIR])
        assert result.ok is True
        assert "122" in capsys.readouterr().out

    def test_reexported_extern_callable_unqualified_through_the_facade(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # utils.ext_facade re-exports utils.ext_math via `export utils.ext_math`
        # (no extern def of its own, so it needs no companion file).
        source = "open import utils/ext_facade\nlet r = double(21)\nprint r\n"
        result = _run_program(source, roots_dirs=[MULTI_FILE_DIR])
        assert result.ok is True
        assert "42" in capsys.readouterr().out

    def test_reexported_extern_callable_through_the_facade_qualifier(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = "open import utils/ext_facade\nlet r = utils/ext_facade::double(21)\nprint r\n"
        result = _run_program(source, roots_dirs=[MULTI_FILE_DIR])
        assert result.ok is True
        assert "42" in capsys.readouterr().out


class TestScopedExecutionFixtures:
    """Scoped imports, opened members, and agents execute across modules."""

    @pytest.mark.parametrize(
        ("responses", "expected"),
        (
            (
                (
                    '{"name": "root", "children": [{"name": "ready", "children": []}]}',
                    '{"$case": "completed"}',
                ),
                "true\ncompleted\ntrue\n",
            ),
            (
                (
                    '{"name": "root", "children": [{"name": "later", "children": []}]}',
                    '{"$case": "waiting"}',
                ),
                "false\nwaiting\nfalse\n",
            ),
        ),
        ids=("completed_task", "waiting_task"),
    )
    def test_path_selected_and_opened_scoped_workflow_executes(
        self,
        responses: tuple[str, str],
        expected: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        entry = MULTI_FILE_DIR / "scoped_execution" / "entry.agl"
        scripted = iter(responses)

        def reviewer(request: Any) -> str:
            del request
            return next(scripted)

        result = _run_program(
            entry.read_text(),
            entry_path=entry,
            roots_dirs=[MULTI_FILE_DIR],
            agents={"reviewer": reviewer},
        )

        assert result.ok is True
        assert capsys.readouterr().out == expected
