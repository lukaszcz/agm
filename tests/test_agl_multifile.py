"""End-to-end tests for AgL multi-file (module-graph) programs.

Tests the graph pipeline: ``prepare_program`` → ``run_prepared_graph``, using
multi-module programs that exercise wildcard imports, qualified imports, library
roots, agent values passed to imported functions, and error diagnostics from
imported modules.

Each test uses ``tmp_path`` for concurrency safety.  Static test programs live
in ``tests/agl/multi_file/`` (outside ``programs/`` to avoid the single-file
e2e harness picking them up).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

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


def _run_graph(
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
        roots=frozenset(
            {*(d.resolve() for d in roots_dirs if d.exists()), REPO_STDLIB_ROOT}
        )
    )
    prepared = PipelineDriver.prepare_program(
        entry_source, entry_path=entry_path, roots=roots
    )
    rt = _make_runtime(default_agent=default_agent)
    if agents:
        for name, fn in agents.items():
            rt.register_agent(name, fn)
    return rt.run_prepared_graph(prepared, param_values=param_values)


# ---------------------------------------------------------------------------
# Scenario 1: wildcard import (import utils.*)
# ---------------------------------------------------------------------------


def test_specific_catch_uses_module_qualified_exception_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A same-named local catch must not catch a distinct imported exception."""
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "lib.agl").write_text(
        "exception Boom extends Exception\n"
        "  detail: text\n"
    )

    source = """\
import lib
exception Boom extends Exception
  code: int
try
  raise lib::Boom(message = "lib", detail = "from lib")
catch Boom =>
  print "wrong"
catch _ =>
  print "ok"
"""
    result = _run_graph(source, roots_dirs=[lib_dir])

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

        source = 'import utils.*\nlet n = add(3, 4)\nlet msg = greet("World")\nprint n\nprint msg\n'
        result = _run_graph(source, roots_dirs=[lib_dir])

        assert result.ok is True
        captured = capsys.readouterr()
        assert "7" in captured.out
        assert "Hello, World!" in captured.out

    def test_wildcard_import_from_repo_fixtures(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Wildcard import: uses multi_file fixture library in tests/agl/multi_file/."""
        entry_file = MULTI_FILE_DIR / "entry_wildcard.agl"
        source = entry_file.read_text()
        result = _run_graph(source, entry_path=entry_file, roots_dirs=[MULTI_FILE_DIR])

        assert result.ok is True
        captured = capsys.readouterr()
        assert "7" in captured.out
        assert "Hello, World!" in captured.out

    def test_wildcard_import_private_name_not_accessible(
        self, tmp_path: Path
    ) -> None:
        """Private names from wildcard-imported modules are not accessible."""
        lib_dir = tmp_path / "lib"
        utils_dir = lib_dir / "utils"
        utils_dir.mkdir(parents=True)
        (utils_dir / "math.agl").write_text(
            "def pub(n: int) -> int = n + 1\n"
            "private def priv(n: int) -> int = n * 2\n"
        )
        # Calling the private name should produce a scope error.
        source = "import utils.*\nlet r = priv(1)\nr\n"
        result = _run_graph(source, roots_dirs=[lib_dir])
        assert result.ok is False
        assert len(result.diagnostics) >= 1


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

        source = "import calc qualified\nlet r = calc::square(5)\nprint r\n"
        result = _run_graph(source, roots_dirs=[lib_dir])

        assert result.ok is True
        captured = capsys.readouterr()
        assert "25" in captured.out

    def test_qualified_import_from_repo_fixtures(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Qualified import: uses multi_file fixture library."""
        entry_file = MULTI_FILE_DIR / "entry_qualified.agl"
        source = entry_file.read_text()
        result = _run_graph(source, entry_path=entry_file, roots_dirs=[MULTI_FILE_DIR])

        assert result.ok is True
        captured = capsys.readouterr()
        assert "25" in captured.out

    def test_qualified_import_unqualified_access_fails(
        self, tmp_path: Path
    ) -> None:
        """Unqualified access to a qualified-only import is a scope error."""
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "calc.agl").write_text("def square(n: int) -> int = n * n\n")

        source = "import calc qualified\nlet r = square(5)\nr\n"
        result = _run_graph(source, roots_dirs=[lib_dir])
        assert result.ok is False


# ---------------------------------------------------------------------------
# Generic inference across module boundaries
# ---------------------------------------------------------------------------


def test_imported_generic_inside_generic_module_body_freshens_per_entry_occurrence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fixtures keep an imported generic use inside app's generic declaration."""
    entry_file = MULTI_FILE_DIR / "entry_inference.agl"
    result = _run_graph(
        entry_file.read_text(), entry_path=entry_file, roots_dirs=[MULTI_FILE_DIR]
    )

    assert result.ok is True
    assert capsys.readouterr().out == "1\ntext\n"


# ---------------------------------------------------------------------------
# Scenario 3: error in imported module reports correct file path
# ---------------------------------------------------------------------------


class TestImportedModuleErrors:
    """Errors in imported modules carry that module's file path."""

    def test_type_error_in_imported_module_names_file(
        self, tmp_path: Path
    ) -> None:
        """A type error in an imported module diagnostic shows that module's path."""
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        bad_mod = lib_dir / "broken.agl"
        # Type error: add(text, text) → int is wrong, but the function body
        # passes a text where int is expected
        bad_mod.write_text('def bad(n: int) -> int = "not an int"\n')

        source = "import broken\nlet r = bad(1)\nprint r\n"
        result = _run_graph(source, roots_dirs=[lib_dir])
        assert result.ok is False
        # At least one diagnostic should mention the broken.agl file
        assert any(
            "broken.agl" in (d.source_label or "")
            for d in result.diagnostics
        ), f"No diagnostic mentions broken.agl; got: {result.diagnostics}"

    def test_scope_error_in_imported_module_names_file(
        self, tmp_path: Path
    ) -> None:
        """A scope error in an imported module diagnostic includes the file label."""
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        bad_mod = lib_dir / "badscope.agl"
        bad_mod.write_text("def f() -> int = undefined_name\n")

        source = "import badscope\nlet r = f()\nr\n"
        result = _run_graph(source, roots_dirs=[lib_dir])
        assert result.ok is False
        assert any(
            "badscope.agl" in (d.source_label or "")
            for d in result.diagnostics
        ), f"No diagnostic mentions badscope.agl; got: {result.diagnostics}"


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
        entry.write_text("import shared\nlet r = double(21)\nprint r\n")

        # Use lib_dir as the lib-root, work_dir as the invocation root.
        source = entry.read_text()
        result = _run_graph(
            source, entry_path=entry, roots_dirs=[work_dir, lib_dir]
        )
        assert result.ok is True
        captured = capsys.readouterr()
        assert "42" in captured.out

    def test_module_not_found_fails(self, tmp_path: Path) -> None:
        """A missing module causes a ModuleNotFound diagnostic."""
        source = "import missing_module\nlet x = 1\nx\n"
        result = _run_graph(source, roots_dirs=[tmp_path])
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

        source = "import shared\nlet r = f()\nr\n"
        result = _run_graph(source, roots_dirs=[root_a, root_b])
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
            "import helper\n"
            'let result = ask_with_agent("test question", mybot)\n'
            "print result\n"
        )

        responses: list[str] = ["mocked answer"]

        def scripted_agent(req: Any) -> str:
            return responses[0]

        result = _run_graph(
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
            "import utils.agent_helper\n"
            'let r = ask_with_agent("ping", mybot)\n'
            "print r\n"
        )

        def scripted_agent(req: Any) -> str:
            return "pong"

        result = _run_graph(
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

        source = "import math\nparam n: int\nlet r = square(n)\nprint r\n"

        from agm.agl import PipelineDriver
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset({lib_dir.resolve(), REPO_STDLIB_ROOT}))
        prepared = PipelineDriver.prepare_program(source, entry_path=None, roots=roots)

        rt = PipelineDriver()
        result = rt.run_prepared_graph(prepared, param_values={"n": 7})
        assert result.ok is True
        captured = capsys.readouterr()
        assert "49" in captured.out

    def test_missing_param_in_multifile_fails(self, tmp_path: Path) -> None:
        """A missing required param in a multi-file program fails cleanly."""
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "calc.agl").write_text("def sq(n: int) -> int = n * n\n")

        source = "import calc\nparam n: int\nlet r = sq(n)\nprint r\n"

        from agm.agl import PipelineDriver
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset({lib_dir.resolve(), REPO_STDLIB_ROOT}))
        prepared = PipelineDriver.prepare_program(source, entry_path=None, roots=roots)

        rt = PipelineDriver()
        result = rt.run_prepared_graph(prepared, param_values={})
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
            "def add(a: int, b: int) -> int = a + b\n"
            "def mul(a: int, b: int) -> int = a * b\n"
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
        # Only 'add' from pkg.math should be in scope; 'mul' must NOT be.
        source = "import pkg.* using add\nlet r = add(3, 4)\nprint r\n"
        result = _run_graph(source, roots_dirs=[lib_dir])
        assert result.ok is True
        captured = capsys.readouterr()
        assert "7" in captured.out

    def test_wildcard_using_hidden_name_inaccessible(self, tmp_path: Path) -> None:
        """Names not listed in 'using' are inaccessible even though exported."""
        lib_dir = self._make_pkg(tmp_path)
        # 'mul' is exported by pkg.math but not listed in using → scope error
        source = "import pkg.* using add\nlet r = mul(3, 4)\nprint r\n"
        result = _run_graph(source, roots_dirs=[lib_dir])
        assert result.ok is False
        assert any("mul" in d.message for d in result.diagnostics)

    def test_wildcard_hiding_removes_name(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """import pkg.* hiding mul brings all names except 'mul' into scope."""
        lib_dir = self._make_pkg(tmp_path)
        # 'add' should still be in scope; 'mul' should not.
        source = "import pkg.* hiding mul\nlet r = add(10, 5)\nprint r\n"
        result = _run_graph(source, roots_dirs=[lib_dir])
        assert result.ok is True
        captured = capsys.readouterr()
        assert "15" in captured.out

    def test_wildcard_hiding_name_inaccessible(self, tmp_path: Path) -> None:
        """The hidden name is inaccessible after hiding."""
        lib_dir = self._make_pkg(tmp_path)
        source = "import pkg.* hiding mul\nlet r = mul(3, 4)\nprint r\n"
        result = _run_graph(source, roots_dirs=[lib_dir])
        assert result.ok is False
        assert any("mul" in d.message for d in result.diagnostics)

    def test_wildcard_using_multi_module_union(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """using selects a name from every matched module that exports it."""
        lib_dir = self._make_pkg(tmp_path)
        # 'add' is in pkg.math and 'greet' is in pkg.text; both should be in scope
        source = (
            "import pkg.* using add, greet\n"
            "let n = add(2, 3)\n"
            'let s = greet("World")\n'
            "print n\n"
            "print s\n"
        )
        result = _run_graph(source, roots_dirs=[lib_dir])
        assert result.ok is True
        captured = capsys.readouterr()
        assert "5" in captured.out
        assert "Hello, World" in captured.out


# ---------------------------------------------------------------------------
# extern def (Python FFI): a library module's extern reachable across
# qualified/open imports, private-extern invisibility, and re-export.
#
# Companion loading, boundary crossing, and the full conversion matrix are
# covered end to end elsewhere (test_agl_extern_runtime.py); this class only
# covers the multi-file import/export surface, using the repo fixture library
# at tests/agl/multi_file/utils/ext_math.agl (+ .py companion) and its
# re-export facade tests/agl/multi_file/utils/ext_facade.agl.
# ---------------------------------------------------------------------------


class TestExternMultiFile:
    """A library module's ``extern def`` through qualified/open imports and re-export."""

    def test_qualified_import_calls_the_extern(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = "import utils.ext_math qualified\nlet r = utils.ext_math::double(21)\nprint r\n"
        result = _run_graph(source, roots_dirs=[MULTI_FILE_DIR])
        assert result.ok is True
        assert "42" in capsys.readouterr().out

    def test_open_import_calls_the_extern_unqualified(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = "import utils.ext_math\nlet r = double(21)\nprint r\n"
        result = _run_graph(source, roots_dirs=[MULTI_FILE_DIR])
        assert result.ok is True
        assert "42" in capsys.readouterr().out

    def test_private_extern_invisible_via_open_import(self) -> None:
        source = "import utils.ext_math\nlet r = secret(21)\nprint r\n"
        result = _run_graph(source, roots_dirs=[MULTI_FILE_DIR])
        assert result.ok is False

    def test_private_extern_invisible_via_qualified_access(self) -> None:
        source = (
            "import utils.ext_math qualified\n"
            "let r = utils.ext_math::secret(21)\n"
            "print r\n"
        )
        result = _run_graph(source, roots_dirs=[MULTI_FILE_DIR])
        assert result.ok is False

    def test_private_extern_usable_from_a_public_function_in_its_own_module(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # `use_secret` (public) calls the private extern `secret` internally —
        # the private extern is only invisible to IMPORTERS, not to code in
        # its own declaring module.
        source = "import utils.ext_math\nlet r = use_secret(21)\nprint r\n"
        result = _run_graph(source, roots_dirs=[MULTI_FILE_DIR])
        assert result.ok is True
        assert "122" in capsys.readouterr().out

    def test_reexported_extern_callable_unqualified_through_the_facade(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # utils.ext_facade re-exports utils.ext_math via `export utils.ext_math`
        # (no extern def of its own, so it needs no companion file).
        source = "import utils.ext_facade\nlet r = double(21)\nprint r\n"
        result = _run_graph(source, roots_dirs=[MULTI_FILE_DIR])
        assert result.ok is True
        assert "42" in capsys.readouterr().out

    def test_reexported_extern_callable_through_the_facade_qualifier(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = (
            "import utils.ext_facade qualified\n"
            "let r = utils.ext_facade::double(21)\n"
            "print r\n"
        )
        result = _run_graph(source, roots_dirs=[MULTI_FILE_DIR])
        assert result.ok is True
        assert "42" in capsys.readouterr().out
