"""Tests for extern (Python FFI) companion loading and pipeline wiring.

Covers everything upstream of the boundary walkers (``test_agl_extern_boundary.py``):
- companion path derivation and existence verification in the module loader.
- ``ExternRegistry`` companion import (once per canonical path) and callable
  resolution, including the misuse contract and failure diagnostics.
- the ``supports_extern`` capability gate.
- pipeline wiring: the registry is built/populated from a checked program's
  extern declarations before evaluation, with fail-fast diagnostics and
  static errors always surfacing before any companion import side effect.

Interpreter dispatch of an extern call is a later stage of this effort, so
tests below that exercise the full pipeline stop at ``check_only`` (static
passes, lowering, and dry-run inventory only) rather than evaluating a
program that calls an extern.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.capabilities import HostCapabilities
from agm.agl.modules.errors import MissingExternCompanion
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.modules.loader import load_graph
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import PipelineDriver, _wire_extern_registry
from agm.agl.runtime.externs import ExternImportError, ExternRegistry, ExternResolutionError
from agm.agl.scope.graph import resolve_graph
from agm.agl.typecheck.graph import CheckedModuleGraph, check_graph
from tests.agl.ir_harness import write_companion_file, write_module_file

_CAPS = HostCapabilities(
    has_default_agent=True,
    supports_shell_exec=True,
    supports_extern=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset(
            {"json", "record", "enum", "list", "dict", "int", "decimal", "bool"}
        ),
    },
)


def _roots(*paths: Path) -> RootSet:
    return RootSet(roots=frozenset(paths))


def _build_checked_graph(
    tmp_path: Path, modules: dict[str, str], capabilities: HostCapabilities = _CAPS
) -> tuple[CheckedModuleGraph, dict[ModuleId, Path | None]]:
    """Load + resolve + check a multi-module graph without lowering or running it.

    Returns the checked graph and the module-id-to-companion-path map the
    loader recorded, mirroring what ``PreparedGraph.companion_paths`` carries.
    """
    root = tmp_path / "root"
    for dotted, source in modules.items():
        if dotted != "entry":
            write_module_file(root, dotted, source)
    graph = load_graph(
        modules.get("entry", "()"), entry_path=None, roots=_roots(root), default_stdlib=False
    )
    resolved = resolve_graph(graph)
    checked = check_graph(resolved, capabilities)
    companion_paths = {mid: lm.companion_path for mid, lm in graph.modules.items()}
    return checked, companion_paths


# ---------------------------------------------------------------------------
# Companion path derivation and existence verification (loader)
# ---------------------------------------------------------------------------


class TestCompanionPathDerivation:
    def test_non_extern_module_has_no_companion_path(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        write_module_file(root, "lib.mod", "def f(x: int) -> int = x")
        graph = load_graph(
            "import lib.mod\n()", entry_path=None, roots=_roots(root), default_stdlib=False
        )
        assert graph.modules[ModuleId.from_dotted("lib.mod")].companion_path is None

    def test_inline_entry_never_needs_a_companion(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        graph = load_graph("()", entry_path=None, roots=_roots(root), default_stdlib=False)
        assert graph.modules[ENTRY_ID].companion_path is None

    def test_extern_module_companion_path_is_the_py_sibling(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        write_module_file(root, "lib.mod", "extern def f(x: int) -> int")
        write_companion_file(root, "lib.mod", "def f(x):\n    return x\n")
        graph = load_graph(
            "import lib.mod\nlib.mod::f(1)",
            entry_path=None,
            roots=_roots(root),
            default_stdlib=False,
        )
        mid = ModuleId.from_dotted("lib.mod")
        assert graph.modules[mid].companion_path == root / "lib" / "mod.py"

    def test_nested_module_directory_companion_path(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        write_module_file(root, "a.b.c", "extern def f(x: int) -> int")
        write_companion_file(root, "a.b.c", "def f(x):\n    return x\n")
        graph = load_graph(
            "import a.b.c\na.b.c::f(1)",
            entry_path=None,
            roots=_roots(root),
            default_stdlib=False,
        )
        mid = ModuleId.from_dotted("a.b.c")
        assert graph.modules[mid].companion_path == root / "a" / "b" / "c.py"

    def test_file_backed_entry_companion_path(self, tmp_path: Path) -> None:
        entry_path = tmp_path / "entry.agl"
        (tmp_path / "entry.py").write_text("def f(x):\n    return x\n")
        graph = load_graph(
            "extern def f(x: int) -> int\n()",
            entry_path=entry_path,
            roots=_roots(tmp_path),
            default_stdlib=False,
        )
        assert graph.modules[ENTRY_ID].companion_path == tmp_path / "entry.py"

    def test_missing_companion_raises_naming_the_module(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        write_module_file(root, "lib.mod", "extern def f(x: int) -> int")
        with pytest.raises(MissingExternCompanion) as excinfo:
            load_graph(
                "import lib.mod\nlib.mod::f(1)",
                entry_path=None,
                roots=_roots(root),
                default_stdlib=False,
            )
        assert "lib.mod" in str(excinfo.value)

    def test_missing_companion_becomes_prepared_graph_diagnostic(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        write_module_file(root, "lib.mod", "extern def f(x: int) -> int")
        prepared = PipelineDriver.prepare_program(
            "import lib.mod\nlib.mod::f(1)",
            entry_path=None,
            roots=_roots(root),
            default_stdlib=False,
        )
        assert prepared.resolved_graph is None
        assert len(prepared.diagnostics) == 1
        assert "lib.mod" in prepared.diagnostics[0].message


# ---------------------------------------------------------------------------
# ExternRegistry: companion import, caching, callable resolution
# ---------------------------------------------------------------------------


class TestExternRegistryLoadAndResolve:
    def test_resolve_returns_the_companion_callable(self, tmp_path: Path) -> None:
        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        mid = ModuleId.from_dotted("lib.mod")
        registry = ExternRegistry()
        registry.load_companion(mid, py_path)
        fn = registry.resolve(mid, "f")
        assert fn(1) == 2

    def test_import_runs_top_level_code_exactly_once_per_registry(self, tmp_path: Path) -> None:
        py_path = tmp_path / "mod.py"
        py_path.write_text("COUNTER = 0\nCOUNTER += 1\ndef f(x):\n    return COUNTER\n")
        mid = ModuleId.from_dotted("lib.mod")
        registry = ExternRegistry()
        registry.load_companion(mid, py_path)
        registry.load_companion(mid, py_path)
        fn = registry.resolve(mid, "f")
        assert fn(None) == 1

    def test_import_is_cached_by_canonical_path_across_module_ids(self, tmp_path: Path) -> None:
        py_path = tmp_path / "mod.py"
        py_path.write_text("COUNTER = 0\nCOUNTER += 1\ndef f(x):\n    return COUNTER\n")
        registry = ExternRegistry()
        registry.load_companion(ModuleId.from_dotted("lib.mod"), py_path)
        registry.load_companion(ModuleId.from_dotted("other.mod"), py_path)
        assert registry.resolve(ModuleId.from_dotted("lib.mod"), "f")(None) == 1
        assert registry.resolve(ModuleId.from_dotted("other.mod"), "f")(None) == 1

    def test_separate_registries_import_independently(self, tmp_path: Path) -> None:
        py_path = tmp_path / "mod.py"
        py_path.write_text("COUNTER = 0\nCOUNTER += 1\ndef f(x):\n    return COUNTER\n")
        mid = ModuleId.from_dotted("lib.mod")
        registry_a = ExternRegistry()
        registry_a.load_companion(mid, py_path)
        registry_b = ExternRegistry()
        registry_b.load_companion(mid, py_path)
        assert registry_a.resolve(mid, "f")(None) == 1
        assert registry_b.resolve(mid, "f")(None) == 1

    def test_missing_attribute_raises_resolution_error(self, tmp_path: Path) -> None:
        py_path = tmp_path / "mod.py"
        py_path.write_text("def wrong_name(x):\n    return x\n")
        mid = ModuleId.from_dotted("lib.mod")
        registry = ExternRegistry()
        registry.load_companion(mid, py_path)
        with pytest.raises(ExternResolutionError) as excinfo:
            registry.resolve(mid, "f")
        assert "lib.mod" in str(excinfo.value)
        assert "'f'" in str(excinfo.value)

    def test_non_callable_attribute_raises_resolution_error(self, tmp_path: Path) -> None:
        py_path = tmp_path / "mod.py"
        py_path.write_text("f = 5\n")
        mid = ModuleId.from_dotted("lib.mod")
        registry = ExternRegistry()
        registry.load_companion(mid, py_path)
        with pytest.raises(ExternResolutionError):
            registry.resolve(mid, "f")

    def test_companion_top_level_exception_raises_import_error(self, tmp_path: Path) -> None:
        py_path = tmp_path / "mod.py"
        py_path.write_text("raise RuntimeError('boom')\n")
        mid = ModuleId.from_dotted("lib.mod")
        registry = ExternRegistry()
        with pytest.raises(ExternImportError) as excinfo:
            registry.load_companion(mid, py_path)
        assert "lib.mod" in str(excinfo.value)

    def test_companion_does_not_pollute_sys_path_or_linger_in_sys_modules(
        self, tmp_path: Path
    ) -> None:
        import sys

        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x\n")
        mid = ModuleId.from_dotted("lib.mod")
        before = set(sys.path)
        registry = ExternRegistry()
        registry.load_companion(mid, py_path)
        assert set(sys.path) == before
        assert not any(name.startswith("agm_agl_extern_companion__") for name in sys.modules)

    def test_resolve_before_load_companion_is_a_programming_error(self) -> None:
        registry = ExternRegistry()
        with pytest.raises(AssertionError):
            registry.resolve(ModuleId.from_dotted("lib.mod"), "f")

    def test_resolve_caches_the_callable_across_repeated_calls(self, tmp_path: Path) -> None:
        """A second ``resolve`` for the same name returns the first lookup's object.

        Swapping the companion module's attribute after the first ``resolve``
        must not change what a later ``resolve`` returns: the cache — not a
        fresh ``getattr`` — is consulted, which also means an extern's
        per-call cost never repeats companion attribute lookup.
        """
        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        mid = ModuleId.from_dotted("lib.mod")
        registry = ExternRegistry()
        module = registry.load_companion(mid, py_path)

        def _replacement(x: int) -> int:
            return x + 999

        first = registry.resolve(mid, "f")
        setattr(module, "f", _replacement)
        second = registry.resolve(mid, "f")

        assert second is first
        assert second(1) == 2


# ---------------------------------------------------------------------------
# Pipeline wiring: capability gate, fail-fast diagnostics, ordering
# ---------------------------------------------------------------------------


class TestCapabilityGate:
    def test_extern_program_rejected_when_capability_off(self, tmp_path: Path) -> None:
        write_companion_file(tmp_path / "root", "lib.mod", "def f(x):\n    return x\n")
        checked, companion_paths = _build_checked_graph(
            tmp_path,
            {
                "entry": "import lib.mod\nlib.mod::f(1)",
                "lib.mod": "extern def f(x: int) -> int",
            },
        )
        caps_off = HostCapabilities(supports_extern=False)
        diagnostics = _wire_extern_registry(
            checked_graph=checked,
            capabilities=caps_off,
            registry=ExternRegistry(),
            companion_paths=companion_paths,
        )
        assert len(diagnostics) == 1

    def test_non_extern_program_unaffected_by_capability_off(self, tmp_path: Path) -> None:
        checked, companion_paths = _build_checked_graph(
            tmp_path, {"entry": "def f(x: int) -> int = x\nf(1)"}
        )
        caps_off = HostCapabilities(supports_extern=False)
        diagnostics = _wire_extern_registry(
            checked_graph=checked,
            capabilities=caps_off,
            registry=ExternRegistry(),
            companion_paths=companion_paths,
        )
        assert diagnostics == []

    def test_extern_program_accepted_when_capability_on(self, tmp_path: Path) -> None:
        write_companion_file(tmp_path / "root", "lib.mod", "def f(x):\n    return x\n")
        checked, companion_paths = _build_checked_graph(
            tmp_path,
            {
                "entry": "import lib.mod\nlib.mod::f(1)",
                "lib.mod": "extern def f(x: int) -> int",
            },
        )
        diagnostics = _wire_extern_registry(
            checked_graph=checked,
            capabilities=_CAPS,
            registry=ExternRegistry(),
            companion_paths=companion_paths,
        )
        assert diagnostics == []


class TestFailFastDiagnostics:
    def test_missing_attribute_diagnostic_names_module_and_function(
        self, tmp_path: Path
    ) -> None:
        write_companion_file(tmp_path / "root", "lib.mod", "def wrong_name(x):\n    return x\n")
        checked, companion_paths = _build_checked_graph(
            tmp_path,
            {
                "entry": "import lib.mod\nlib.mod::f(1)",
                "lib.mod": "extern def f(x: int) -> int",
            },
        )
        diagnostics = _wire_extern_registry(
            checked_graph=checked,
            capabilities=_CAPS,
            registry=ExternRegistry(),
            companion_paths=companion_paths,
        )
        assert len(diagnostics) == 1
        assert "lib.mod" in diagnostics[0].message
        assert "'f'" in diagnostics[0].message

    def test_non_callable_attribute_is_a_diagnostic(self, tmp_path: Path) -> None:
        write_companion_file(tmp_path / "root", "lib.mod", "f = 5\n")
        checked, companion_paths = _build_checked_graph(
            tmp_path,
            {
                "entry": "import lib.mod\nlib.mod::f(1)",
                "lib.mod": "extern def f(x: int) -> int",
            },
        )
        diagnostics = _wire_extern_registry(
            checked_graph=checked,
            capabilities=_CAPS,
            registry=ExternRegistry(),
            companion_paths=companion_paths,
        )
        assert len(diagnostics) == 1

    def test_a_module_with_several_externs_imports_its_companion_only_once(
        self, tmp_path: Path
    ) -> None:
        write_companion_file(
            tmp_path / "root",
            "lib.mod",
            "COUNTER = 0\nCOUNTER += 1\n"
            "def f(x):\n    return COUNTER\n"
            "def g(x):\n    return COUNTER\n",
        )
        checked, companion_paths = _build_checked_graph(
            tmp_path,
            {
                "entry": "import lib.mod\nlib.mod::f(1)\nlib.mod::g(1)",
                "lib.mod": "extern def f(x: int) -> int\nextern def g(x: int) -> int",
            },
        )
        registry = ExternRegistry()
        diagnostics = _wire_extern_registry(
            checked_graph=checked,
            capabilities=_CAPS,
            registry=registry,
            companion_paths=companion_paths,
        )
        assert diagnostics == []
        mid = ModuleId.from_dotted("lib.mod")
        assert registry.resolve(mid, "f")(None) == 1
        assert registry.resolve(mid, "g")(None) == 1

    def test_one_import_failure_reports_once_for_every_extern_in_that_module(
        self, tmp_path: Path
    ) -> None:
        write_companion_file(tmp_path / "root", "lib.mod", "raise RuntimeError('boom')\n")
        checked, companion_paths = _build_checked_graph(
            tmp_path,
            {
                "entry": "import lib.mod\nlib.mod::f(1)\nlib.mod::g(1)",
                "lib.mod": "extern def f(x: int) -> int\nextern def g(x: int) -> int",
            },
        )
        diagnostics = _wire_extern_registry(
            checked_graph=checked,
            capabilities=_CAPS,
            registry=ExternRegistry(),
            companion_paths=companion_paths,
        )
        assert len(diagnostics) == 1

    def test_diagnostics_via_run_prepared_graph_before_lowering(self, tmp_path: Path) -> None:
        """The same wiring runs from the real pipeline entry point, cleanly (no crash)."""
        write_module_file(tmp_path / "root", "lib.mod", "extern def f(x: int) -> int")
        write_companion_file(tmp_path / "root", "lib.mod", "def wrong_name(x):\n    return x\n")
        driver = PipelineDriver()
        prepared = PipelineDriver.prepare_program(
            "import lib.mod\nlib.mod::f(1)",
            entry_path=None,
            roots=_roots(tmp_path / "root"),
            default_stdlib=False,
        )
        result = driver.run_prepared_graph(prepared)
        assert result.ok is False
        assert len(result.diagnostics) == 1
        assert "lib.mod" in result.diagnostics[0].message
        assert "'f'" in result.diagnostics[0].message


class TestOrdering:
    def test_type_error_reported_before_any_companion_import(self, tmp_path: Path) -> None:
        marker = tmp_path / "marker.txt"
        write_module_file(tmp_path / "root", "lib.mod", "extern def f(x: int) -> int")
        write_companion_file(
            tmp_path / "root",
            "lib.mod",
            f"open({str(marker)!r}, 'w').write('imported')\n"
            "def wrong_name(x):\n    return x\n",
        )
        driver = PipelineDriver()
        prepared = PipelineDriver.prepare_program(
            'import lib.mod\n1 + "a"',
            entry_path=None,
            roots=_roots(tmp_path / "root"),
            default_stdlib=False,
        )
        result = driver.run_prepared_graph(prepared)
        assert result.ok is False
        assert any("operand" in d.message.lower() for d in result.diagnostics)
        assert not marker.exists()


class TestRegistryPopulatedViaPipeline:
    def test_registry_resolves_every_declared_extern_before_evaluation(
        self, tmp_path: Path
    ) -> None:
        write_module_file(tmp_path / "root", "lib.mod", "extern def f(x: int) -> int")
        write_companion_file(tmp_path / "root", "lib.mod", "def f(x):\n    return x + 1\n")
        driver = PipelineDriver()
        prepared = PipelineDriver.prepare_program(
            "import lib.mod\nlib.mod::f(1)",
            entry_path=None,
            roots=_roots(tmp_path / "root"),
            default_stdlib=False,
        )
        discovery = driver.discover_params_graph(prepared)
        assert discovery.checked_graph is not None

        diagnostics = _wire_extern_registry(
            checked_graph=discovery.checked_graph,
            capabilities=driver.host_environment().capabilities,
            registry=driver.host_environment().extern_registry,
            companion_paths=prepared.companion_paths,
        )
        assert diagnostics == []
        fn = driver.host_environment().extern_registry.resolve(
            ModuleId.from_dotted("lib.mod"), "f"
        )
        assert fn(1) == 2

    def test_run_prepared_graph_wires_the_registry_before_lowering_and_lowers_cleanly(
        self, tmp_path: Path
    ) -> None:
        """``run_prepared_graph`` itself performs the wiring, not just the helper.

        Interpreter dispatch of an extern call is a later stage of this effort,
        so this only drives the pipeline through ``check_only`` (static passes,
        lowering, and dry-run inventory — no evaluation) to prove the registry
        is fully populated and lowering succeeds, without depending on
        evaluating the extern call itself.
        """
        write_module_file(tmp_path / "root", "lib.mod", "extern def f(x: int) -> int")
        write_companion_file(tmp_path / "root", "lib.mod", "def f(x):\n    return x + 1\n")
        driver = PipelineDriver()
        prepared = PipelineDriver.prepare_program(
            "import lib.mod\nlib.mod::f(1)",
            entry_path=None,
            roots=_roots(tmp_path / "root"),
            default_stdlib=False,
        )
        result = driver.run_prepared_graph(prepared, check_only=True)
        assert result.ok is True
        assert [cs.callee for cs in result.call_sites] == ["f"]
        fn = driver.host_environment().extern_registry.resolve(
            ModuleId.from_dotted("lib.mod"), "f"
        )
        assert fn(1) == 2
