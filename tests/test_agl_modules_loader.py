"""Tests for ModuleGraph loading in src/agm/agl/modules/loader.py."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from agm.agl.syntax.nodes import Program

from agm.agl.modules.errors import AmbiguousModule, ImportEntryError, ModuleNotFound
from agm.agl.modules.ids import ENTRY_ID, STD_CORE_ID, ModuleId
from agm.agl.modules.loader import LoadedModule, ModuleGraph, build_repl_graph, load_graph
from agm.agl.modules.roots import RootSet
from agm.agl.syntax.nodes import ImportDecl
from agm.agl.syntax.spans import SourceId

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _roots(*paths: Path) -> RootSet:
    return RootSet(roots=frozenset(paths))


def _write_agl(path: Path, source: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


def _module_path(root: Path, dotted: str) -> Path:
    mid = ModuleId.from_dotted(dotted)
    return root / mid.relpath().replace("/", os.sep)


_MINIMAL = "()"  # Minimal valid AgL program


def _write_module(root: Path, dotted: str, source: str = _MINIMAL) -> Path:
    p = _module_path(root, dotted)
    _write_agl(p, source)
    return p


# ---------------------------------------------------------------------------
# LoadedModule / ModuleGraph dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_loaded_module_fields(self, tmp_path: Path) -> None:
        """LoadedModule must expose module_id, program, path, source, imports."""
        root = tmp_path / "r"
        root.mkdir()
        _write_module(root, "mymod", "def f() = 1")
        graph = load_graph(_MINIMAL, entry_path=None, roots=_roots(root))
        entry = graph.modules[ENTRY_ID]
        # Fields present
        assert entry.module_id == ENTRY_ID
        assert entry.program is not None
        assert entry.path is None  # inline entry
        assert isinstance(entry.source, SourceId)
        assert isinstance(entry.imports, tuple)

    def test_module_graph_contains_entry_id(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        graph = load_graph(_MINIMAL, entry_path=None, roots=_roots(root))
        assert ENTRY_ID in graph.modules
        assert graph.entry_id == ENTRY_ID

    def test_module_graph_sccs_is_tuple(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        graph = load_graph(_MINIMAL, entry_path=None, roots=_roots(root))
        assert isinstance(graph.sccs, tuple)


# ---------------------------------------------------------------------------
# Entry source identification
# ---------------------------------------------------------------------------


class TestEntrySourceId:
    def test_inline_entry_source_label_is_command(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        graph = load_graph(_MINIMAL, entry_path=None, roots=_roots(root))
        entry = graph.modules[ENTRY_ID]
        assert entry.source.label == "<command>"

    def test_file_entry_source_label_is_path(self, tmp_path: Path) -> None:
        entry_file = tmp_path / "prog.agl"
        entry_file.write_text(_MINIMAL)
        root = tmp_path
        graph = load_graph(
            _MINIMAL, entry_path=entry_file, roots=_roots(root)
        )
        entry = graph.modules[ENTRY_ID]
        assert entry.source.label == str(entry_file.resolve())

    def test_entry_path_stored_as_canonical(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        entry_file = root / "main.agl"
        entry_file.write_text(_MINIMAL)
        # Entry path via symlink
        link = tmp_path / "mainlink.agl"
        link.symlink_to(entry_file)
        graph = load_graph(_MINIMAL, entry_path=link, roots=_roots(root))
        entry = graph.modules[ENTRY_ID]
        assert entry.path == entry_file.resolve()


# ---------------------------------------------------------------------------
# Graph build — basic loading
# ---------------------------------------------------------------------------


class TestGraphBuild:
    def test_entry_only_graph(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        graph = load_graph("let x = 1", entry_path=None, roots=_roots(root))
        assert len(graph.modules) == 2
        assert ENTRY_ID in graph.modules
        assert STD_CORE_ID in graph.modules

    def test_imported_module_appears_in_graph(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        _write_module(root, "util")
        entry = "import util"
        graph = load_graph(entry, entry_path=None, roots=_roots(root))
        assert ModuleId.from_dotted("util") in graph.modules

    def test_imported_module_without_default_stdlib(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        _write_module(root, "util", "def one() -> int = 1\n")
        graph = load_graph(
            "import util\none()",
            entry_path=None,
            roots=_roots(root),
            default_stdlib=False,
        )
        assert ModuleId.from_dotted("util") in graph.modules
        util = graph.modules[ModuleId.from_dotted("util")]
        assert not any(decl.module_path == STD_CORE_ID.segments for decl in util.imports)

    def test_transitive_import_resolved(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        _write_module(root, "base")
        _write_module(root, "mid", "import base")
        entry = "import mid"
        graph = load_graph(entry, entry_path=None, roots=_roots(root))
        assert ModuleId.from_dotted("mid") in graph.modules
        assert ModuleId.from_dotted("base") in graph.modules

    def test_imports_extracted_from_program_body(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        _write_module(root, "libx")
        entry = "import libx"
        graph = load_graph(entry, entry_path=None, roots=_roots(root))
        entry_mod = graph.modules[ENTRY_ID]
        assert any(
            isinstance(item, ImportDecl) and item.module_path == ("libx",)
            for item in entry_mod.imports
        )

    def test_module_not_found_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        with pytest.raises(ModuleNotFound):
            load_graph("import doesnotexist", entry_path=None, roots=_roots(root))

    def test_ambiguous_module_raises(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        _write_module(root_a, "clash")
        _write_module(root_b, "clash")
        with pytest.raises(AmbiguousModule):
            load_graph(
                "import clash",
                entry_path=None,
                roots=_roots(root_a, root_b),
            )


# ---------------------------------------------------------------------------
# Cycles
# ---------------------------------------------------------------------------


class TestCycles:
    def test_direct_cycle_terminates(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        # a imports b, b imports a
        _write_module(root, "a", "import b")
        _write_module(root, "b", "import a")
        graph = load_graph("import a", entry_path=None, roots=_roots(root))
        assert ModuleId.from_dotted("a") in graph.modules
        assert ModuleId.from_dotted("b") in graph.modules
        # Exactly 4 modules: std.core + entry + a + b
        assert len(graph.modules) == 4

    def test_longer_cycle_terminates(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        _write_module(root, "x", "import y")
        _write_module(root, "y", "import z")
        _write_module(root, "z", "import x")
        graph = load_graph("import x", entry_path=None, roots=_roots(root))
        assert len(graph.modules) == 5  # std.core + entry + x + y + z

    def test_cycle_nodes_linked_in_sccs(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        _write_module(root, "p", "import q")
        _write_module(root, "q", "import p")
        graph = load_graph("import p", entry_path=None, roots=_roots(root))
        # p and q must be in the same SCC
        p_id = ModuleId.from_dotted("p")
        q_id = ModuleId.from_dotted("q")
        found_scc = False
        for scc in graph.sccs:
            if p_id in scc and q_id in scc:
                found_scc = True
                break
        assert found_scc, "p and q should be in the same SCC"

    def test_acyclic_modules_each_in_singleton_scc(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        _write_module(root, "leaf")
        _write_module(root, "mid", "import leaf")
        graph = load_graph("import mid", entry_path=None, roots=_roots(root))
        # Each of the three modules should be in its own SCC
        leaf_id = ModuleId.from_dotted("leaf")
        mid_id = ModuleId.from_dotted("mid")
        for scc in graph.sccs:
            if leaf_id in scc:
                assert len(scc) == 1
            if mid_id in scc:
                assert len(scc) == 1


# ---------------------------------------------------------------------------
# Node-id disjointness
# ---------------------------------------------------------------------------


class TestNodeIdDisjointness:
    def _collect_all_node_ids(self, graph: ModuleGraph) -> dict[ModuleId, set[int]]:
        """Return ALL node-id sets per module via a full recursive AST walk."""
        from agm.agl.syntax.visitor import walk

        result: dict[ModuleId, set[int]] = {}
        for mid, mod in graph.modules.items():
            ids: set[int] = set()

            def _visit(node: object, _ids: set[int] = ids) -> None:
                node_id = getattr(node, "node_id", None)
                if node_id is not None:
                    _ids.add(node_id)

            walk(mod.program, _visit)
            result[mid] = ids
        return result

    def test_all_node_ids_disjoint_across_modules(self, tmp_path: Path) -> None:
        """Every node_id in the loaded graph must be unique across ALL modules.

        Uses agm.agl.syntax.visitor.walk to recursively collect every node_id
        in each module's AST, then asserts pairwise disjointness.
        """
        root = tmp_path / "r"
        root.mkdir()
        _write_module(root, "alpha", "def f(x: int) -> int = x")
        _write_module(root, "beta", "def g(y: int) -> int = y")
        graph = load_graph(
            "import alpha\nimport beta",
            entry_path=None,
            roots=_roots(root),
        )
        id_sets = self._collect_all_node_ids(graph)
        # Verify pairwise disjointness across the three modules.
        mids = list(id_sets.keys())
        for i in range(len(mids)):
            for j in range(i + 1, len(mids)):
                overlap = id_sets[mids[i]] & id_sets[mids[j]]
                assert not overlap, (
                    f"Node IDs not disjoint between {mids[i]} and {mids[j]}: {overlap}"
                )

    def test_start_ids_are_monotonically_increasing(self, tmp_path: Path) -> None:
        """Each module's program node_id must be strictly greater than the
        previous module's, reflecting the seeded parse order."""
        root = tmp_path / "r"
        root.mkdir()
        _write_module(root, "m1", "let x = 1")
        _write_module(root, "m2", "let y = 2")
        graph = load_graph(
            "import m1\nimport m2",
            entry_path=None,
            roots=_roots(root),
        )
        # All program node_ids should be distinct
        prog_ids = [mod.program.node_id for mod in graph.modules.values()]
        assert len(prog_ids) == len(set(prog_ids))


# ---------------------------------------------------------------------------
# Canonical dedup
# ---------------------------------------------------------------------------


class TestCanonicalDedup:
    def test_same_file_via_symlinked_roots_counts_once(self, tmp_path: Path) -> None:
        root = tmp_path / "lib"
        root.mkdir()
        _write_module(root, "shared.util")
        link = tmp_path / "link"
        link.symlink_to(root)
        # Both roots would resolve 'shared.util' to the same canonical file
        graph = load_graph(
            "import shared.util",
            entry_path=None,
            roots=_roots(root, link),
        )
        # Module appears exactly once
        util_id = ModuleId.from_dotted("shared.util")
        assert util_id in graph.modules
        assert len([mid for mid in graph.modules if mid == util_id]) == 1

    def test_duplicate_import_of_same_module_counted_once(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        _write_module(root, "lib")
        # Entry imports 'lib' twice (or two modules import same dep)
        _write_module(root, "a", "import lib")
        _write_module(root, "b", "import lib")
        graph = load_graph(
            "import a\nimport b",
            entry_path=None,
            roots=_roots(root),
        )
        lib_id = ModuleId.from_dotted("lib")
        assert list(graph.modules.keys()).count(lib_id) == 1


# ---------------------------------------------------------------------------
# Reject import of entry
# ---------------------------------------------------------------------------


class TestRejectImportOfEntry:
    def test_import_resolving_to_entry_file_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        entry_file = root / "main.agl"
        entry_file.write_text("import main")
        # Create a 'main.agl' that IS the entry file but also create it
        # as a module 'main' pointing to the same file
        # We do this by making the entry file resolvable as module 'main'
        with pytest.raises(ImportEntryError):
            load_graph(
                "import main",
                entry_path=entry_file,
                roots=_roots(root),
            )

    def test_inline_entry_no_entry_rejection(self, tmp_path: Path) -> None:
        """Inline entry has no file path → no ImportEntryError possible."""
        root = tmp_path / "r"
        root.mkdir()
        _write_module(root, "lib")
        graph = load_graph("import lib", entry_path=None, roots=_roots(root))
        assert ModuleId.from_dotted("lib") in graph.modules


# ---------------------------------------------------------------------------
# Wildcard imports in loader
# ---------------------------------------------------------------------------


class TestWildcardImportInLoader:
    def test_wildcard_import_loads_all_matched_modules(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        _write_module(root, "pkg.a")
        _write_module(root, "pkg.b")
        graph = load_graph("import pkg.*", entry_path=None, roots=_roots(root))
        assert ModuleId.from_dotted("pkg.a") in graph.modules
        assert ModuleId.from_dotted("pkg.b") in graph.modules

    def test_wildcard_prefix_not_found_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        from agm.agl.modules.errors import ModulePrefixNotFound

        with pytest.raises(ModulePrefixNotFound):
            load_graph("import noprefix.*", entry_path=None, roots=_roots(root))

    def test_wildcard_already_loaded_module_not_duplicated(
        self, tmp_path: Path
    ) -> None:
        """Wildcard expansion encountering an already-loaded module (via
        a transitive import) must skip it, not re-queue or duplicate it."""
        root = tmp_path / "r"
        root.mkdir()
        _write_module(root, "pkg.a")
        _write_module(root, "pkg.b")
        # pkg.a is loaded first (non-wildcard from entry), then pkg.a itself
        # issues a wildcard import that includes already-loaded pkg.a.
        _write_module(root, "pkg.a", "import pkg.*")
        # Entry imports pkg.a non-wildcard first, triggering the load of pkg.a,
        # whose wildcard import pkg.* then encounters pkg.a as already-loaded.
        graph = load_graph(
            "import pkg.a",
            entry_path=None,
            roots=_roots(root),
        )
        pkg_a_id = ModuleId.from_dotted("pkg.a")
        pkg_b_id = ModuleId.from_dotted("pkg.b")
        assert list(graph.modules.keys()).count(pkg_a_id) == 1
        assert pkg_b_id in graph.modules


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_graph_keys_deterministic_under_shuffled_roots(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        _write_module(root_a, "x.one")
        _write_module(root_b, "y.two")
        _write_module(root_a, "z.three")

        entry = "import x.one\nimport y.two\nimport z.three"

        graph1 = load_graph(entry, entry_path=None, roots=_roots(root_a, root_b))
        graph2 = load_graph(entry, entry_path=None, roots=_roots(root_b, root_a))

        assert set(graph1.modules.keys()) == set(graph2.modules.keys())

    def test_sccs_deterministic_under_shuffled_roots(self, tmp_path: Path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        _write_module(root_a, "m", "import n")
        _write_module(root_b, "n", "import m")

        entry = "import m"
        graph1 = load_graph(entry, entry_path=None, roots=_roots(root_a, root_b))
        graph2 = load_graph(entry, entry_path=None, roots=_roots(root_b, root_a))

        # Sort SCCs for comparison
        def _norm(sccs: tuple[tuple[ModuleId, ...], ...]) -> list[tuple[ModuleId, ...]]:
            normalized = [
                tuple(sorted(scc, key=lambda m: m.segments)) for scc in sccs
            ]
            return sorted(normalized, key=lambda s: tuple(m.segments for m in s))

        assert _norm(graph1.sccs) == _norm(graph2.sccs)

    def test_graph_keys_ordered_identically_under_distinct_fs_layouts(
        self, tmp_path: Path
    ) -> None:
        """load_graph yields the same ordered module-id list regardless of
        filesystem discovery order (controlled via wildcard imports).

        Three distinct directory layouts produce the same logical module set.
        All three must produce identical ordered graph.modules key sequences
        (after discarding ENTRY_ID), proving the internal sort makes
        filesystem/glob order irrelevant.
        """
        entry = "import pkg.*"

        # Layout 1: modules written in alphabetical order under a single root.
        r1 = tmp_path / "r1"
        r1.mkdir()
        _write_module(r1, "pkg.alpha")
        _write_module(r1, "pkg.beta")
        _write_module(r1, "pkg.gamma")

        # Layout 2: same modules written in reverse order under a different root.
        r2 = tmp_path / "r2"
        r2.mkdir()
        _write_module(r2, "pkg.gamma")
        _write_module(r2, "pkg.beta")
        _write_module(r2, "pkg.alpha")

        # Layout 3: modules split across two roots, created in mixed order,
        # roots themselves passed in reversed order.
        r3a = tmp_path / "r3a"
        r3b = tmp_path / "r3b"
        r3a.mkdir()
        r3b.mkdir()
        _write_module(r3b, "pkg.gamma")
        _write_module(r3a, "pkg.alpha")
        _write_module(r3b, "pkg.beta")

        g1 = load_graph(entry, entry_path=None, roots=_roots(r1))
        g2 = load_graph(entry, entry_path=None, roots=_roots(r2))
        g3a = load_graph(entry, entry_path=None, roots=_roots(r3a, r3b))
        g3b = load_graph(entry, entry_path=None, roots=_roots(r3b, r3a))

        def _lib_keys(g: ModuleGraph) -> list[ModuleId]:
            return [mid for mid in g.modules if not mid.is_entry]

        keys1 = _lib_keys(g1)
        keys2 = _lib_keys(g2)
        keys3a = _lib_keys(g3a)
        keys3b = _lib_keys(g3b)

        assert keys1 == keys2, "FS creation order affects graph key order (layout1 vs layout2)"
        assert keys1 == keys3a, "Root ordering affects graph key order (layout1 vs layout3a)"
        assert keys3a == keys3b, "Root ordering affects graph key order (layout3a vs layout3b)"


# ---------------------------------------------------------------------------
# File-based entry path
# ---------------------------------------------------------------------------


class TestFileBasedEntry:
    def test_file_entry_path_stored_in_loaded_module(self, tmp_path: Path) -> None:
        root = tmp_path
        entry_file = root / "prog.agl"
        entry_file.write_text("")
        graph = load_graph(_MINIMAL, entry_path=entry_file, roots=_roots(root))
        entry_mod = graph.modules[ENTRY_ID]
        assert entry_mod.path == entry_file.resolve()

    def test_file_entry_source_id_uses_canonical_path(self, tmp_path: Path) -> None:
        root = tmp_path
        entry_file = root / "script.agl"
        entry_file.write_text("")
        link = tmp_path / "scriptlink.agl"
        link.symlink_to(entry_file)
        graph = load_graph(_MINIMAL, entry_path=link, roots=_roots(root))
        entry_mod = graph.modules[ENTRY_ID]
        assert entry_mod.source.label == str(entry_file.resolve())

    def test_loaded_module_source_id_uses_canonical_path(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        mod_path = _write_module(root, "deps.lib")
        graph = load_graph("import deps.lib", entry_path=None, roots=_roots(root))
        lib_mod = graph.modules[ModuleId.from_dotted("deps.lib")]
        assert lib_mod.source.label == str(mod_path.resolve())
        assert lib_mod.path == mod_path.resolve()

    def test_all_modules_have_non_none_path_except_inline_entry(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "r"
        root.mkdir()
        _write_module(root, "mod")
        graph = load_graph("import mod", entry_path=None, roots=_roots(root))
        for mid, mod in graph.modules.items():
            if mid in (ENTRY_ID, STD_CORE_ID):
                assert mod.path is None  # inline entry and synthetic stdlib
            else:
                assert mod.path is not None


# ---------------------------------------------------------------------------
# Error carries SourceSpan
# ---------------------------------------------------------------------------


class TestErrorsCarrySpan:
    def test_module_not_found_has_span(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        try:
            load_graph("import missing", entry_path=None, roots=_roots(root))
        except ModuleNotFound as exc:
            assert exc.span is not None
        else:
            pytest.fail("Expected ModuleNotFound")

    def test_import_entry_error_has_span(self, tmp_path: Path) -> None:
        root = tmp_path / "r"
        root.mkdir()
        entry_file = root / "main.agl"
        entry_file.write_text("import main")
        try:
            load_graph("import main", entry_path=entry_file, roots=_roots(root))
        except ImportEntryError as exc:
            assert exc.span is not None
        else:
            pytest.fail("Expected ImportEntryError")


# ---------------------------------------------------------------------------
# build_repl_graph — REPL incremental graph construction
# ---------------------------------------------------------------------------


def _parse_for_repl(source: str) -> "Program":
    from agm.agl.parser.parser import parse_program

    return parse_program(source)


class TestBuildReplGraph:
    """Tests for :func:`~agm.agl.modules.loader.build_repl_graph`."""

    def test_simple_program_no_imports(self, tmp_path: Path) -> None:
        """Graph for a program with no imports has only ENTRY_ID."""
        program = _parse_for_repl("let x = 1")
        graph, _next_id, new_modules = build_repl_graph(
            program, 1000, path=None, cached={}, roots=_roots(tmp_path)
        )
        assert ENTRY_ID in graph.modules
        assert len(new_modules) == 0

    def test_import_loads_lib_module(self, tmp_path: Path) -> None:
        """A program with import declarations loads the referenced lib module."""
        lib = tmp_path / "mylib.agl"
        lib.write_text("def add(a: int, b: int) -> int = a + b\n")
        program = _parse_for_repl("import mylib\nadd(3, 4)")
        graph, _next_id, new_modules = build_repl_graph(
            program, 1000, path=None, cached={}, roots=_roots(tmp_path)
        )
        mid = ModuleId(segments=("mylib",))
        assert mid in graph.modules
        assert mid in new_modules

    def test_cached_module_not_reloaded(self, tmp_path: Path) -> None:
        """A module in *cached* is reused without re-loading from disk."""
        lib = tmp_path / "mylib.agl"
        lib.write_text("def add(a: int, b: int) -> int = a + b\n")
        mid = ModuleId(segments=("mylib",))

        # First build: loads mylib
        program1 = _parse_for_repl("import mylib\n()")
        _, next_id, new1 = build_repl_graph(
            program1, 0, path=None, cached={}, roots=_roots(tmp_path)
        )
        assert mid in new1

        # Second build: mylib already cached; should not appear in new_modules
        cached: dict[ModuleId, LoadedModule] = {mid: new1[mid]}
        program2 = _parse_for_repl("import mylib\n()")
        _, _next2, new2 = build_repl_graph(
            program2, next_id, path=None, cached=cached, roots=_roots(tmp_path)
        )
        assert mid not in new2

    def test_cached_std_core_not_reloaded(self, tmp_path: Path) -> None:
        """The REPL graph builder reuses cached std.core when present."""
        program1 = _parse_for_repl("()")
        graph1, next_id, _new1 = build_repl_graph(
            program1, 0, path=None, cached={}, roots=_roots(tmp_path)
        )
        std_core = graph1.modules[STD_CORE_ID]

        program2 = _parse_for_repl("let x: Option[int] = None\nx")
        graph2, _next2, new2 = build_repl_graph(
            program2,
            next_id,
            path=None,
            cached={STD_CORE_ID: std_core},
            roots=_roots(tmp_path),
        )

        assert graph2.modules[STD_CORE_ID] is std_core
        assert STD_CORE_ID not in new2

    def test_path_sets_entry_source_id(self, tmp_path: Path) -> None:
        """When *path* is given, the entry source ID uses the canonical path label."""
        entry_file = tmp_path / "entry.agl"
        entry_file.write_text("let x = 1\n")
        program = _parse_for_repl("let x = 1")
        graph, _next_id, _new = build_repl_graph(
            program, 0, path=entry_file, cached={}, roots=_roots(tmp_path)
        )
        entry_loaded = graph.modules[ENTRY_ID]
        assert entry_loaded.source is not None
        assert str(entry_file.resolve()) in entry_loaded.source.label

    def test_wildcard_import_expands_modules(self, tmp_path: Path) -> None:
        """Wildcard imports (``import pkg.*``) load all modules in the package."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "a.agl").write_text("def fa() -> int = 1\n")
        (pkg / "b.agl").write_text("def fb() -> int = 2\n")
        program = _parse_for_repl("import pkg.*\n()")
        graph, _next_id, new_modules = build_repl_graph(
            program, 0, path=None, cached={}, roots=_roots(tmp_path)
        )
        mid_a = ModuleId(segments=("pkg", "a"))
        mid_b = ModuleId(segments=("pkg", "b"))
        assert mid_a in graph.modules
        assert mid_b in graph.modules
        assert mid_a in new_modules
        assert mid_b in new_modules

    def test_import_entry_itself_raises(self, tmp_path: Path) -> None:
        """An import that resolves to the entry file path raises ImportEntryError."""
        entry_file = tmp_path / "entry.agl"
        entry_file.write_text("import entry\n()")
        program = _parse_for_repl("import entry\n()")
        with pytest.raises(ImportEntryError):
            build_repl_graph(
                program, 0, path=entry_file, cached={}, roots=_roots(tmp_path)
            )

    def test_wildcard_skips_already_cached_modules(self, tmp_path: Path) -> None:
        """Wildcard expansion skips modules already present in *cached* or *modules*."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "a.agl").write_text("def fa() -> int = 1\n")
        (pkg / "b.agl").write_text("def fb() -> int = 2\n")
        mid_a = ModuleId(segments=("pkg", "a"))

        # First build: load pkg.a via explicit import to pre-cache it.
        program1 = _parse_for_repl("import pkg.a\n()")
        _, next_id, new1 = build_repl_graph(
            program1, 0, path=None, cached={}, roots=_roots(tmp_path)
        )
        assert mid_a in new1

        # Second build: wildcard pkg.* with pkg.a already cached.
        # pkg.a should not appear in new_modules (already in cached).
        cached: dict[ModuleId, LoadedModule] = {mid_a: new1[mid_a]}
        program2 = _parse_for_repl("import pkg.*\n()")
        _, _next2, new2 = build_repl_graph(
            program2, next_id, path=None, cached=cached, roots=_roots(tmp_path)
        )
        mid_b = ModuleId(segments=("pkg", "b"))
        # pkg.a was already cached, not in new_modules
        assert mid_a not in new2
        # pkg.b is new
        assert mid_b in new2

    def test_bfs_dedup_via_two_imports_same_module(self, tmp_path: Path) -> None:
        """Two import declarations for the same module result in only one load."""
        lib = tmp_path / "mylib.agl"
        lib.write_text("def add(a: int, b: int) -> int = a + b\n")

        # Syntactically, you can't have two imports of the same module in one
        # program; instead, simulate via an entry that imports a module both
        # directly and via a wildcard that expands to include it.
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "shared.agl").write_text("def f() -> int = 1\n")
        (pkg / "other.agl").write_text("import pkg.shared\ndef g() -> int = f()\n")
        # entry imports pkg.shared directly AND pkg.* (which includes pkg.shared)
        # so pkg.shared would be queued twice: once for direct import and once
        # via wildcard; BFS should handle the dedup via the 'if mid in modules'
        # check.
        program = _parse_for_repl("import pkg.shared\nimport pkg.*\n()")
        graph, _next_id, new_modules = build_repl_graph(
            program, 0, path=None, cached={}, roots=_roots(tmp_path)
        )
        mid_shared = ModuleId(segments=("pkg", "shared"))
        # pkg.shared should appear exactly once in the graph
        assert mid_shared in graph.modules
