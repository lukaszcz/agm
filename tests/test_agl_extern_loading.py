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

import os
import py_compile
import sys
import time
from pathlib import Path
from types import CodeType

import pytest

from agm.agl import artifact_storage
from agm.agl.capabilities import HostCapabilities
from agm.agl.diagnostics import Diagnostic
from agm.agl.modules.errors import MissingExternCompanion
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import PipelineDriver, _wire_extern_registry
from agm.agl.runtime import externs
from agm.agl.runtime.externs import (
    ExternImportError,
    ExternRegistry,
    ExternResolutionError,
    _CompanionBytecodeLoader,
)
from agm.agl.scope.program import resolve_program
from agm.agl.typecheck.program import CheckedProgram, check_program
from agm.core import fs
from tests._agl_helpers import file_program, prepare_inline_command
from tests.agl.ir_harness import age_file, write_companion_file, write_module_file
from tests.agl.module_graph import load_graph

_CAPS = HostCapabilities(
    supports_shell_exec=True,
    supports_extern=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}),
    },
)


# Message of the contract diagnostic this module injects; owned by the test, so
# asserting on it pins propagation rather than production wording.
_INJECTED_CONTRACT_DIAGNOSTIC = "injected contract failure"


def _roots(*paths: Path) -> RootSet:
    return RootSet(roots=frozenset(paths))


def _build_checked(
    tmp_path: Path, modules: dict[str, str], capabilities: HostCapabilities = _CAPS
) -> tuple[CheckedProgram, dict[ModuleId, Path | None]]:
    """Load + resolve + check a multi-module graph without lowering or running it.

    Returns the checked graph and the module-id-to-companion-path map the
    loader recorded, mirroring what ``PreparedProgram.companion_paths`` carries.
    """
    root = tmp_path / "root"
    for module_path, source in modules.items():
        if module_path != "entry":
            write_module_file(root, module_path, source)
    graph = load_graph(
        file_program(modules.get("entry", "()")),
        entry_path=None,
        roots=_roots(root),
        default_stdlib=False,
    )
    resolved = resolve_program(graph)
    checked = check_program(resolved, capabilities)
    companion_paths = {mid: lm.companion_path for mid, lm in graph.modules.items()}
    return checked, companion_paths


# ---------------------------------------------------------------------------
# Companion path derivation and existence verification (loader)
# ---------------------------------------------------------------------------


class TestCompanionPathDerivation:
    def test_non_extern_module_has_no_companion_path(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        write_module_file(root, "lib/mod", "def f(x: int) -> int = x")
        graph = load_graph(
            "import lib/mod::*\n()", entry_path=None, roots=_roots(root), default_stdlib=False
        )
        assert graph.modules[ModuleId.from_path("lib/mod")].companion_path is None

    def test_inline_entry_never_needs_a_companion(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        graph = load_graph("()", entry_path=None, roots=_roots(root), default_stdlib=False)
        assert graph.modules[ENTRY_ID].companion_path is None

    def test_extern_module_companion_path_is_the_py_sibling(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        write_module_file(root, "lib/mod", "extern def f(x: int) -> int")
        write_companion_file(root, "lib/mod", "def f(x):\n    return x\n")
        graph = load_graph(
            "import lib/mod::*\nlib/mod::f(1)",
            entry_path=None,
            roots=_roots(root),
            default_stdlib=False,
        )
        mid = ModuleId.from_path("lib/mod")
        assert graph.modules[mid].companion_path == root / "lib" / "mod.py"

    def test_nested_module_directory_companion_path(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        write_module_file(root, "a/b/c", "extern def f(x: int) -> int")
        write_companion_file(root, "a/b/c", "def f(x):\n    return x\n")
        graph = load_graph(
            "import a/b/c::*\na/b/c::f(1)",
            entry_path=None,
            roots=_roots(root),
            default_stdlib=False,
        )
        mid = ModuleId.from_path("a/b/c")
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

    @pytest.mark.parametrize(
        "source",
        (
            "scope Group\n  extern def f(x: int) -> int\nend Group\n\n()",
            "extern def Group::f(x: int) -> int\n()",
        ),
        ids=("region", "shorthand"),
    )
    def test_scoped_externs_use_the_module_companion(self, tmp_path: Path, source: str) -> None:
        companion = tmp_path / "entry.py"
        companion.write_text("def f(x):\n    return x\n")
        graph = load_graph(
            source,
            entry_path=tmp_path / "entry.agl",
            roots=_roots(tmp_path),
            default_stdlib=False,
        )

        assert graph.modules[ENTRY_ID].companion_path == companion

        result = PipelineDriver().run(
            file_program(source),
            entry_path=tmp_path / "entry.agl",
            roots=_roots(tmp_path),
            default_stdlib=False,
        )
        assert result.ok

    def test_missing_companion_raises_naming_the_module(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        write_module_file(root, "lib/mod", "extern def f(x: int) -> int")
        with pytest.raises(MissingExternCompanion) as excinfo:
            load_graph(
                "import lib/mod::*\nlib/mod::f(1)",
                entry_path=None,
                roots=_roots(root),
                default_stdlib=False,
            )
        assert "lib/mod" in str(excinfo.value)

    def test_missing_companion_becomes_prepared_program_diagnostic(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        write_module_file(root, "lib/mod", "extern def f(x: int) -> int")
        prepared = prepare_inline_command(
            "import lib/mod::*\nlib/mod::f(1)",
            roots=_roots(root),
            default_stdlib=False,
        )
        assert prepared.resolved is None
        assert len(prepared.diagnostics) == 1
        assert "lib/mod" in prepared.diagnostics[0].message


# ---------------------------------------------------------------------------
# ExternRegistry: companion import, caching, callable resolution
# ---------------------------------------------------------------------------


class TestExternRegistryLoadAndResolve:
    def test_resolve_returns_the_companion_callable(self, tmp_path: Path) -> None:
        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        mid = ModuleId.from_path("lib/mod")
        registry = ExternRegistry()
        registry.load_companion(mid, py_path)
        fn = registry.resolve(mid, "f")
        assert fn(1) == 2

    def test_import_writes_no_bytecode_cache_beside_the_companion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An installed package's companion lives in the immutable store, which
        # AGM must leave exactly as its RECORD describes it.
        monkeypatch.setattr(sys, "dont_write_bytecode", False)
        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        mid = ModuleId.from_path("lib/mod")
        registry = ExternRegistry()

        registry.load_companion(mid, py_path)

        assert registry.resolve(mid, "f")(1) == 2
        assert sorted(path.name for path in tmp_path.iterdir()) == ["mod.py"]
        assert not list(tmp_path.rglob("*.pyc"))

    def test_import_runs_top_level_code_exactly_once_per_registry(self, tmp_path: Path) -> None:
        py_path = tmp_path / "mod.py"
        py_path.write_text("COUNTER = 0\nCOUNTER += 1\ndef f(x):\n    return COUNTER\n")
        mid = ModuleId.from_path("lib/mod")
        registry = ExternRegistry()
        registry.load_companion(mid, py_path)
        registry.load_companion(mid, py_path)
        fn = registry.resolve(mid, "f")
        assert fn(None) == 1

    def test_import_is_cached_by_canonical_path_across_module_ids(self, tmp_path: Path) -> None:
        py_path = tmp_path / "mod.py"
        py_path.write_text("COUNTER = 0\nCOUNTER += 1\ndef f(x):\n    return COUNTER\n")
        registry = ExternRegistry()
        registry.load_companion(ModuleId.from_path("lib/mod"), py_path)
        registry.load_companion(ModuleId.from_path("other/mod"), py_path)
        assert registry.resolve(ModuleId.from_path("lib/mod"), "f")(None) == 1
        assert registry.resolve(ModuleId.from_path("other/mod"), "f")(None) == 1

    def test_separate_registries_import_independently(self, tmp_path: Path) -> None:
        py_path = tmp_path / "mod.py"
        py_path.write_text("COUNTER = 0\nCOUNTER += 1\ndef f(x):\n    return COUNTER\n")
        mid = ModuleId.from_path("lib/mod")
        registry_a = ExternRegistry()
        registry_a.load_companion(mid, py_path)
        registry_b = ExternRegistry()
        registry_b.load_companion(mid, py_path)
        assert registry_a.resolve(mid, "f")(None) == 1
        assert registry_b.resolve(mid, "f")(None) == 1

    def test_a_rewritten_companion_is_reimported(self, tmp_path: Path) -> None:
        """An edited companion runs its new code, not the code first imported.

        The registry caches by canonical path so that two module ids sharing
        one companion do not run its top level twice. Without a validity
        check that cache also outlives the file: a REPL session that edits a
        companion, or a host that replaces one, would keep calling the old
        Python for the life of the registry, with nothing to say why.
        """
        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        mid = ModuleId.from_path("lib/mod")
        registry = ExternRegistry()
        registry.load_companion(mid, py_path)
        assert registry.resolve(mid, "f")(1) == 2

        py_path.write_text("def f(x):\n    return x + 100\n")
        registry.load_companion(mid, py_path)

        assert registry.resolve(mid, "f")(1) == 101

    def test_a_rewritten_companion_of_the_same_size_is_reimported(self, tmp_path: Path) -> None:
        """Size alone cannot decide staleness, so the stamp carries the mtime too."""
        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        mid = ModuleId.from_path("lib/mod")
        registry = ExternRegistry()
        registry.load_companion(mid, py_path)
        assert registry.resolve(mid, "f")(1) == 2

        replacement = "def f(x):\n    return x + 9\n"
        assert len(replacement) == len("def f(x):\n    return x + 1\n")
        py_path.write_text(replacement)
        # Set the timestamp rather than trusting the clock: a filesystem with
        # coarse timestamps could otherwise stamp both writes identically and
        # make this pass or fail on granularity instead of on the rule.
        stamp = py_path.stat().st_mtime_ns + 1_000_000_000
        os.utime(py_path, ns=(stamp, stamp))
        registry.load_companion(mid, py_path)

        assert registry.resolve(mid, "f")(1) == 10

    def test_reimport_ignores_a_timestamp_valid_stale_bytecode_cache(self, tmp_path: Path) -> None:
        """A changed companion is compiled from source even when its pyc looks current."""
        original = "def f(x):\n    return x + 1\n"
        replacement = "def f(x):\n    return x + 9\n"
        assert len(replacement) == len(original)
        py_path = tmp_path / "mod.py"
        py_path.write_text(original)
        second = py_path.stat().st_mtime_ns // 1_000_000_000
        original_mtime = second * 1_000_000_000 + 100_000_000
        os.utime(py_path, ns=(original_mtime, original_mtime))
        bytecode_path = py_compile.compile(str(py_path), doraise=True)
        assert Path(bytecode_path).is_file()

        mid = ModuleId.from_path("lib/mod")
        registry = ExternRegistry()
        registry.load_companion(mid, py_path)
        assert registry.resolve(mid, "f")(1) == 2

        py_path.write_text(replacement)
        replacement_mtime = second * 1_000_000_000 + 200_000_000
        os.utime(py_path, ns=(replacement_mtime, replacement_mtime))
        registry.load_companion(mid, py_path)

        assert registry.resolve(mid, "f")(1) == 10

    def test_missing_attribute_raises_resolution_error(self, tmp_path: Path) -> None:
        py_path = tmp_path / "mod.py"
        py_path.write_text("def wrong_name(x):\n    return x\n")
        mid = ModuleId.from_path("lib/mod")
        registry = ExternRegistry()
        registry.load_companion(mid, py_path)
        with pytest.raises(ExternResolutionError) as excinfo:
            registry.resolve(mid, "f")
        assert "lib/mod" in str(excinfo.value)
        assert "'f'" in str(excinfo.value)

    def test_non_callable_attribute_raises_resolution_error(self, tmp_path: Path) -> None:
        py_path = tmp_path / "mod.py"
        py_path.write_text("f = 5\n")
        mid = ModuleId.from_path("lib/mod")
        registry = ExternRegistry()
        registry.load_companion(mid, py_path)
        with pytest.raises(ExternResolutionError):
            registry.resolve(mid, "f")

    def test_companion_top_level_exception_raises_import_error(self, tmp_path: Path) -> None:
        py_path = tmp_path / "mod.py"
        py_path.write_text("raise RuntimeError('boom')\n")
        mid = ModuleId.from_path("lib/mod")
        registry = ExternRegistry()
        with pytest.raises(ExternImportError) as excinfo:
            registry.load_companion(mid, py_path)
        assert "lib/mod" in str(excinfo.value)

    def test_companion_importing_option_helpers_without_std_option_fails_to_load(
        self, tmp_path: Path
    ) -> None:
        """``agl.option_none``/``option_some`` exist only once ``std/option``'s
        ``Option`` is among the registered nominals; without it the ``agl``
        module simply has no such attribute, so the companion's own import
        fails like any other missing name."""
        py_path = tmp_path / "mod.py"
        py_path.write_text("from agl import option_none\n")
        mid = ModuleId.from_path("lib/mod")
        registry = ExternRegistry()
        with pytest.raises(ExternImportError):
            registry.load_companion(mid, py_path)

    def test_companion_does_not_pollute_sys_path_or_linger_in_sys_modules(
        self, tmp_path: Path
    ) -> None:
        import sys

        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x\n")
        mid = ModuleId.from_path("lib/mod")
        before = set(sys.path)
        registry = ExternRegistry()
        registry.load_companion(mid, py_path)
        assert set(sys.path) == before
        assert not any(name.startswith("agm_agl_extern_companion__") for name in sys.modules)
        assert "agl" not in sys.modules

    def test_a_companion_that_disappeared_is_reported_as_an_import_failure(
        self, tmp_path: Path
    ) -> None:
        """A vanished companion is a diagnosable import failure, not a crash.

        The loader records a companion path when it discovers the file; by the
        time the extern is loaded the file may be gone. The pipeline turns an
        import failure into a diagnostic, so anything else escapes as an
        unhandled error out of a command.
        """
        registry = ExternRegistry()

        with pytest.raises(ExternImportError):
            registry.load_companion(ModuleId.from_path("lib/mod"), tmp_path / "gone.py")

    def test_failed_companion_import_also_removes_agl_module(self, tmp_path: Path) -> None:
        import sys

        py_path = tmp_path / "mod.py"
        py_path.write_text("raise RuntimeError('boom')\n")
        registry = ExternRegistry()

        with pytest.raises(ExternImportError):
            registry.load_companion(ModuleId.from_path("lib/mod"), py_path)

        assert "agl" not in sys.modules

    def test_companion_import_restores_an_existing_agl_module(self, tmp_path: Path) -> None:
        import sys
        from types import ModuleType

        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x): return x\n")
        existing = ModuleType("agl")
        previous = sys.modules.get("agl")
        sys.modules["agl"] = existing
        try:
            ExternRegistry().load_companion(ModuleId.from_path("lib/mod"), py_path)
            assert sys.modules["agl"] is existing
        finally:
            if previous is None:
                sys.modules.pop("agl", None)
            else:
                sys.modules["agl"] = previous

    def test_resolve_before_load_companion_is_a_programming_error(self) -> None:
        registry = ExternRegistry()
        with pytest.raises(AssertionError):
            registry.resolve(ModuleId.from_path("lib/mod"), "f")

    def test_resolve_caches_the_callable_across_repeated_calls(self, tmp_path: Path) -> None:
        """A second ``resolve`` for the same name returns the first lookup's object.

        Swapping the companion module's attribute after the first ``resolve``
        must not change what a later ``resolve`` returns: the cache — not a
        fresh ``getattr`` — is consulted, which also means an extern's
        per-call cost never repeats companion attribute lookup.
        """
        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        mid = ModuleId.from_path("lib/mod")
        registry = ExternRegistry()
        module = registry.load_companion(mid, py_path)

        def _replacement(x: int) -> int:
            return x + 999

        first = registry.resolve(mid, "f")
        setattr(module, "f", _replacement)
        second = registry.resolve(mid, "f")

        assert second is first
        assert second(1) == 2

    def test_remapping_module_id_invalidates_resolved_callables(self, tmp_path: Path) -> None:
        old_path = tmp_path / "old.py"
        new_path = tmp_path / "new.py"
        old_path.write_text("def f(x):\n    return 'old'\n")
        new_path.write_text("def f(x):\n    return 'new'\n")
        mid = ModuleId.from_path("lib/mod")
        registry = ExternRegistry()

        registry.load_companion(mid, old_path)
        assert registry.resolve(mid, "f")(None) == "old"

        registry.load_companion(mid, new_path)
        assert registry.resolve(mid, "f")(None) == "new"


# ---------------------------------------------------------------------------
# Companion bytecode cache: read/write of the AgL artifact cache
# ---------------------------------------------------------------------------


def _count_source_to_code(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Patch the loader to delegate to its real compiler while counting calls."""
    original = _CompanionBytecodeLoader.source_to_code
    counts = [0]

    def _tracking(loader: _CompanionBytecodeLoader, data: bytes, path: str) -> CodeType:
        counts[0] += 1
        return original(loader, data, path)

    monkeypatch.setattr(_CompanionBytecodeLoader, "source_to_code", _tracking)
    return counts


class TestCompanionBytecodeCache:
    def test_second_load_from_a_fresh_registry_is_served_from_the_bytecode_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fresh registry (a new process, in effect) skips compilation on a hit."""
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        age_file(py_path)
        mid = ModuleId.from_path("lib/mod")
        ExternRegistry().load_companion(mid, py_path)

        def _unexpected(loader: _CompanionBytecodeLoader, data: bytes, path: str) -> CodeType:
            raise AssertionError("companion should be served from the bytecode cache")

        monkeypatch.setattr(_CompanionBytecodeLoader, "source_to_code", _unexpected)

        registry = ExternRegistry()
        registry.load_companion(mid, py_path)
        assert registry.resolve(mid, "f")(1) == 2

    def test_companion_changed_after_cache_read_is_compiled_from_current_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cache hit is invalid when the companion changes before it is used."""
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        age_file(py_path)
        mid = ModuleId.from_path("lib/mod")
        ExternRegistry().load_companion(mid, py_path)

        original_read_payload = externs.read_payload

        def _rewrite_after_cache_read(entry: Path, identity: bytes) -> bytes | None:
            payload = original_read_payload(entry, identity)
            py_path.write_text("def f(x):\n    return x + 9\n")
            age_file(py_path)
            return payload

        monkeypatch.setattr(externs, "read_payload", _rewrite_after_cache_read)

        registry = ExternRegistry()
        registry.load_companion(mid, py_path)

        assert registry.resolve(mid, "f")(1) == 10

    def test_companion_removed_after_cache_read_fails_to_import(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cache hit cannot import a companion deleted before it is used."""
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        age_file(py_path)
        mid = ModuleId.from_path("lib/mod")
        ExternRegistry().load_companion(mid, py_path)

        original_read_payload = externs.read_payload

        def _remove_after_cache_read(entry: Path, identity: bytes) -> bytes | None:
            payload = original_read_payload(entry, identity)
            py_path.unlink()
            return payload

        monkeypatch.setattr(externs, "read_payload", _remove_after_cache_read)

        with pytest.raises(ExternImportError):
            ExternRegistry().load_companion(mid, py_path)

    def test_companion_edited_in_place_without_changing_its_stamp_keeps_serving_cached_code(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The stamp, not the source, defines cache validity: this is that contract.

        An edit that preserves size, mtime, and inode (written in place, with the
        mtime restored) is indistinguishable from no edit at all, so a fresh
        registry -- a new process, in effect -- keeps serving the code cached
        under that stamp.
        """
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        age_file(py_path)
        mid = ModuleId.from_path("lib/mod")
        ExternRegistry().load_companion(mid, py_path)
        stamp_before = fs.identity_stamp(py_path)

        replacement = "def f(x):\n    return x + 9\n"
        assert len(replacement) == len("def f(x):\n    return x + 1\n")
        py_path.write_text(replacement)
        os.utime(py_path, ns=(stamp_before[0], stamp_before[0]))
        assert fs.identity_stamp(py_path) == stamp_before

        registry = ExternRegistry()
        registry.load_companion(mid, py_path)
        assert registry.resolve(mid, "f")(1) == 2

    def test_damaged_pyc_cache_entry_falls_back_to_compiling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = tmp_path / "cache"
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        age_file(py_path)
        mid = ModuleId.from_path("lib/mod")
        ExternRegistry().load_companion(mid, py_path)

        entries = list((cache / "agm" / "agl").glob("*.pyc"))
        assert entries
        for entry in entries:
            entry.write_bytes(b"damaged")

        registry = ExternRegistry()
        registry.load_companion(mid, py_path)
        assert registry.resolve(mid, "f")(1) == 2

    def test_pyc_entry_that_fails_to_unmarshal_falls_back_to_compiling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A payload passing the storage envelope's integrity check but not marshalled code.

        Only the ``marshal.loads`` guard catches this kind of miss.
        """
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        age_file(py_path)
        mid = ModuleId.from_path("lib/mod")

        monkeypatch.setattr(externs, "read_payload", lambda entry, identity: b"not marshalled code")
        counts = _count_source_to_code(monkeypatch)

        registry = ExternRegistry()
        registry.load_companion(mid, py_path)

        assert counts[0] == 1
        assert registry.resolve(mid, "f")(1) == 2

    def test_unwritable_cache_home_falls_back_to_compiling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = tmp_path / "cache"
        cache.write_text("a file cannot hold a cache directory")
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        age_file(py_path)
        mid = ModuleId.from_path("lib/mod")
        registry = ExternRegistry()
        registry.load_companion(mid, py_path)
        assert registry.resolve(mid, "f")(1) == 2

    def test_entry_written_under_a_different_compiler_digest_is_a_miss(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        age_file(py_path)
        mid = ModuleId.from_path("lib/mod")
        ExternRegistry().load_companion(mid, py_path)

        monkeypatch.setattr(artifact_storage, "_COMPILER_DIGEST", b"\x00" * 32)
        counts = _count_source_to_code(monkeypatch)

        registry = ExternRegistry()
        registry.load_companion(mid, py_path)
        assert counts[0] == 1
        assert registry.resolve(mid, "f")(1) == 2

    def test_bytecode_cached_at_one_optimize_level_is_not_served_at_another(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cache key covers ``-O``, so one level's cached code never serves another."""
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        age_file(py_path)
        mid = ModuleId.from_path("lib/mod")

        monkeypatch.setattr(externs, "_optimize_level", lambda: 0)
        ExternRegistry().load_companion(mid, py_path)

        monkeypatch.setattr(externs, "_optimize_level", lambda: 2)
        counts = _count_source_to_code(monkeypatch)
        registry = ExternRegistry()
        registry.load_companion(mid, py_path)

        assert counts[0] == 1
        assert registry.resolve(mid, "f")(1) == 2

    def test_reimport_after_a_changed_stamp_keeps_one_bytecode_entry_per_companion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The slot is stable per companion: a stamp change overwrites it, not duplicates it."""
        cache = tmp_path / "cache"
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        age_file(py_path)
        mid = ModuleId.from_path("lib/mod")
        registry = ExternRegistry()
        registry.load_companion(mid, py_path)

        # Aged to a distinctly different offset than the first version, so the
        # two stamps differ even on a filesystem with coarse mtime granularity.
        py_path.write_text("def f(x):\n    return x + 9\n")
        age_file(py_path, seconds=7200)
        registry.load_companion(mid, py_path)

        assert registry.resolve(mid, "f")(1) == 10
        assert len(list((cache / "agm" / "agl").glob("*.pyc"))) == 1

    def test_freshly_written_companion_is_compiled_but_not_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stamp that could still collide with a same-tick edit is never published.

        The mtime is pinned to the future rather than left at "now": that
        keeps the assertion true regardless of how long compiling this test's
        companion takes, instead of racing the 2-second settle window.
        """
        cache = tmp_path / "cache"
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        future = time.time_ns() + 3_600_000_000_000
        os.utime(py_path, ns=(future, future))
        mid = ModuleId.from_path("lib/mod")

        registry = ExternRegistry()
        registry.load_companion(mid, py_path)

        assert registry.resolve(mid, "f")(1) == 2
        assert not list((cache / "agm" / "agl").glob("*.pyc"))

    def test_companion_changed_between_the_registry_stamp_and_the_read_is_not_cached_stale(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file that moves on between the registry's stamp and the compile is never published.

        The registry stamps the file, then the loader reads it -- an edit
        landing in that window must not let code compiled from the new source
        be published under the stamp taken before it.
        """
        cache = tmp_path / "cache"
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        age_file(py_path)
        mid = ModuleId.from_path("lib/mod")

        original_get_data = _CompanionBytecodeLoader.get_data

        def _mutate_before_read(loader: _CompanionBytecodeLoader, path: str) -> bytes:
            py_path.write_text("def f(x):\n    return x + 999\n")
            age_file(py_path)
            return original_get_data(loader, path)

        monkeypatch.setattr(_CompanionBytecodeLoader, "get_data", _mutate_before_read)

        registry = ExternRegistry()
        registry.load_companion(mid, py_path)

        assert registry.resolve(mid, "f")(1) == 1000
        assert not list((cache / "agm" / "agl").glob("*.pyc"))

    def test_companion_removed_after_compiling_still_loads_and_is_not_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cache must never be the reason an import fails.

        A companion that vanishes between the compile and the post-compile
        re-stat must still load and run -- the write is simply skipped.
        """
        cache = tmp_path / "cache"
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
        py_path = tmp_path / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        age_file(py_path)
        mid = ModuleId.from_path("lib/mod")

        original_get_data = _CompanionBytecodeLoader.get_data

        def _vanish_after_read(loader: _CompanionBytecodeLoader, path: str) -> bytes:
            data = original_get_data(loader, path)
            py_path.unlink()
            return data

        monkeypatch.setattr(_CompanionBytecodeLoader, "get_data", _vanish_after_read)

        registry = ExternRegistry()
        registry.load_companion(mid, py_path)

        assert registry.resolve(mid, "f")(1) == 2
        assert not list((cache / "agm" / "agl").glob("*.pyc"))

    def test_companion_under_a_non_utf8_directory_name_loads_and_runs(self, tmp_path: Path) -> None:
        """A cache key built from a non-UTF-8 path must not crash a loadable companion."""
        try:
            bad_dir = tmp_path / os.fsdecode(b"bad\xffdir")
            bad_dir.mkdir()
        except OSError:
            pytest.skip("filesystem rejects a non-UTF-8 directory name")
        py_path = bad_dir / "mod.py"
        py_path.write_text("def f(x):\n    return x + 1\n")
        mid = ModuleId.from_path("lib/mod")
        registry = ExternRegistry()
        registry.load_companion(mid, py_path)
        assert registry.resolve(mid, "f")(1) == 2


# ---------------------------------------------------------------------------
# Pipeline wiring: capability gate, fail-fast diagnostics, ordering
# ---------------------------------------------------------------------------


class TestCapabilityGate:
    def test_extern_program_rejected_when_capability_off(self, tmp_path: Path) -> None:
        write_companion_file(tmp_path / "root", "lib/mod", "def f(x):\n    return x\n")
        checked, companion_paths = _build_checked(
            tmp_path,
            {
                "entry": "import lib/mod::*\nlib/mod::f(1)",
                "lib/mod": "extern def f(x: int) -> int",
            },
        )
        caps_off = HostCapabilities(supports_extern=False)
        diagnostics = _wire_extern_registry(
            checked=checked,
            capabilities=caps_off,
            registry=ExternRegistry(),
            companion_paths=companion_paths,
        )
        assert len(diagnostics) == 1

    def test_non_extern_program_unaffected_by_capability_off(self, tmp_path: Path) -> None:
        checked, companion_paths = _build_checked(
            tmp_path, {"entry": "def f(x: int) -> int = x\nf(1)"}
        )
        caps_off = HostCapabilities(supports_extern=False)
        diagnostics = _wire_extern_registry(
            checked=checked,
            capabilities=caps_off,
            registry=ExternRegistry(),
            companion_paths=companion_paths,
        )
        assert diagnostics == []

    def test_extern_program_accepted_when_capability_on(self, tmp_path: Path) -> None:
        write_companion_file(tmp_path / "root", "lib/mod", "def f(x):\n    return x\n")
        checked, companion_paths = _build_checked(
            tmp_path,
            {
                "entry": "import lib/mod::*\nlib/mod::f(1)",
                "lib/mod": "extern def f(x: int) -> int",
            },
        )
        diagnostics = _wire_extern_registry(
            checked=checked,
            capabilities=_CAPS,
            registry=ExternRegistry(),
            companion_paths=companion_paths,
        )
        assert diagnostics == []


class TestFailFastDiagnostics:
    def test_missing_attribute_diagnostic_names_module_and_function(self, tmp_path: Path) -> None:
        write_companion_file(tmp_path / "root", "lib/mod", "def wrong_name(x):\n    return x\n")
        checked, companion_paths = _build_checked(
            tmp_path,
            {
                "entry": "import lib/mod::*\nlib/mod::f(1)",
                "lib/mod": "extern def f(x: int) -> int",
            },
        )
        diagnostics = _wire_extern_registry(
            checked=checked,
            capabilities=_CAPS,
            registry=ExternRegistry(),
            companion_paths=companion_paths,
        )
        assert len(diagnostics) == 1
        assert "lib/mod" in diagnostics[0].message
        assert "'f'" in diagnostics[0].message

    def test_non_callable_attribute_is_a_diagnostic(self, tmp_path: Path) -> None:
        write_companion_file(tmp_path / "root", "lib/mod", "f = 5\n")
        checked, companion_paths = _build_checked(
            tmp_path,
            {
                "entry": "import lib/mod::*\nlib/mod::f(1)",
                "lib/mod": "extern def f(x: int) -> int",
            },
        )
        diagnostics = _wire_extern_registry(
            checked=checked,
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
            "lib/mod",
            "COUNTER = 0\nCOUNTER += 1\n"
            "def f(x):\n    return COUNTER\n"
            "def g(x):\n    return COUNTER\n",
        )
        checked, companion_paths = _build_checked(
            tmp_path,
            {
                "entry": "import lib/mod::*\nlet _ = lib/mod::f(1)\nlet _ = lib/mod::g(1)",
                "lib/mod": "extern def f(x: int) -> int\nextern def g(x: int) -> int",
            },
        )
        registry = ExternRegistry()
        diagnostics = _wire_extern_registry(
            checked=checked,
            capabilities=_CAPS,
            registry=registry,
            companion_paths=companion_paths,
        )
        assert diagnostics == []
        mid = ModuleId.from_path("lib/mod")
        assert registry.resolve(mid, "f")(None) == 1
        assert registry.resolve(mid, "g")(None) == 1

    def test_one_import_failure_reports_once_for_every_extern_in_that_module(
        self, tmp_path: Path
    ) -> None:
        write_companion_file(tmp_path / "root", "lib/mod", "raise RuntimeError('boom')\n")
        checked, companion_paths = _build_checked(
            tmp_path,
            {
                "entry": "import lib/mod::*\nlet _ = lib/mod::f(1)\nlet _ = lib/mod::g(1)",
                "lib/mod": "extern def f(x: int) -> int\nextern def g(x: int) -> int",
            },
        )
        diagnostics = _wire_extern_registry(
            checked=checked,
            capabilities=_CAPS,
            registry=ExternRegistry(),
            companion_paths=companion_paths,
        )
        assert len(diagnostics) == 1

    def test_diagnostics_via_run_prepared_before_lowering(self, tmp_path: Path) -> None:
        """The same wiring runs from the real pipeline entry point, cleanly (no crash)."""
        write_module_file(tmp_path / "root", "lib/mod", "extern def f(x: int) -> int")
        write_companion_file(tmp_path / "root", "lib/mod", "def wrong_name(x):\n    return x\n")
        driver = PipelineDriver()
        prepared = prepare_inline_command(
            "import lib/mod::*\nlib/mod::f(1)",
            roots=_roots(tmp_path / "root"),
            default_stdlib=False,
        )
        result = driver.run_prepared(prepared)
        assert result.ok is False
        assert len(result.diagnostics) == 1
        assert "lib/mod" in result.diagnostics[0].message
        assert "'f'" in result.diagnostics[0].message


class TestOrdering:
    def test_type_error_reported_before_any_companion_import(self, tmp_path: Path) -> None:
        marker = tmp_path / "marker.txt"
        write_module_file(tmp_path / "root", "lib/mod", "extern def f(x: int) -> int")
        write_companion_file(
            tmp_path / "root",
            "lib/mod",
            f"open({str(marker)!r}, 'w').write('imported')\ndef wrong_name(x):\n    return x\n",
        )
        driver = PipelineDriver()
        prepared = prepare_inline_command(
            'import lib/mod::*\n1 + "a"',
            roots=_roots(tmp_path / "root"),
            default_stdlib=False,
        )
        result = driver.run_prepared(prepared)
        assert result.ok is False
        assert result.diagnostics
        # The type error is what was reported: nothing complains about the
        # extern module, whose companion was never imported.
        assert all("lib/mod" not in d.message for d in result.diagnostics)
        assert not marker.exists()

    def test_custom_contract_error_reported_before_any_companion_import(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        marker = tmp_path / "marker.txt"
        write_module_file(tmp_path / "root", "lib/mod", "extern def f(x: int) -> int")
        write_companion_file(
            tmp_path / "root",
            "lib/mod",
            f"open({str(marker)!r}, 'w').write('imported')\ndef f(x):\n    return x\n",
        )
        monkeypatch.setattr(
            "agm.agl.pipeline._materialize_program_custom_contract_payloads",
            lambda checked, codecs: (
                {},
                [Diagnostic(message=_INJECTED_CONTRACT_DIAGNOSTIC, line=1)],
            ),
        )
        driver = PipelineDriver()
        prepared = prepare_inline_command(
            "import lib/mod::*\nlib/mod::f(1)",
            roots=_roots(tmp_path / "root"),
            default_stdlib=False,
        )
        result = driver.run_prepared(prepared)
        assert result.ok is False
        assert any(d.message == _INJECTED_CONTRACT_DIAGNOSTIC for d in result.diagnostics)
        assert not marker.exists()


class TestRegistryPopulatedViaPipeline:
    def test_registry_resolves_every_declared_extern_before_evaluation(
        self, tmp_path: Path
    ) -> None:
        write_module_file(tmp_path / "root", "lib/mod", "extern def f(x: int) -> int")
        write_companion_file(tmp_path / "root", "lib/mod", "def f(x):\n    return x + 1\n")
        driver = PipelineDriver()
        prepared = prepare_inline_command(
            "import lib/mod::*\nlib/mod::f(1)",
            roots=_roots(tmp_path / "root"),
            default_stdlib=False,
        )
        discovery = driver.discover_programs(prepared)
        assert discovery.checked is not None

        diagnostics = _wire_extern_registry(
            checked=discovery.checked,
            capabilities=driver.host_environment().capabilities,
            registry=driver.host_environment().extern_registry,
            companion_paths=prepared.companion_paths,
        )
        assert diagnostics == []
        fn = driver.host_environment().extern_registry.resolve(ModuleId.from_path("lib/mod"), "f")
        assert fn(1) == 2

    def test_run_prepared_wires_the_registry_before_evaluation(self, tmp_path: Path) -> None:
        """``run_prepared`` itself performs real-run extern wiring."""
        write_module_file(tmp_path / "root", "lib/mod", "extern def f(x: int) -> int")
        write_companion_file(tmp_path / "root", "lib/mod", "def f(x):\n    return x + 1\n")
        driver = PipelineDriver()
        prepared = prepare_inline_command(
            "import lib/mod::*\nlib/mod::f(1)",
            roots=_roots(tmp_path / "root"),
            default_stdlib=False,
        )
        result = driver.run_prepared(prepared)
        assert result.ok is True
        fn = driver.host_environment().extern_registry.resolve(ModuleId.from_path("lib/mod"), "f")
        assert fn(1) == 2
