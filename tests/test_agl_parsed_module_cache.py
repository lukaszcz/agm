"""Tests for the process-global parsed standard-library module cache."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from agm.agl.modules import loader as loader_module
from agm.agl.modules.errors import MissingExternCompanion
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.modules.loader import LoadedModule, load_graph
from agm.agl.modules.parsed_module_cache import (
    RESERVED_NODE_ID_BASE,
    ModuleDerivationCache,
    ParsedModuleCache,
    clear_parsed_module_cache,
)
from agm.agl.modules.roots import RootSet
from agm.agl.pipeline import PipelineDriver
from tests._agl_helpers import all_node_ids, prepare_inline_command, run_inline_command

_LIB_ID = ModuleId.from_path("lib/a")
_ENTRY_SOURCE = "import lib/a\n"
_LIB_SOURCE = "def f() -> int = 1\n"


@pytest.fixture(autouse=True)
def _isolated_cache() -> Iterator[None]:
    """Give every test a cold cache and leave no entries behind."""
    clear_parsed_module_cache()
    yield
    clear_parsed_module_cache()


@pytest.fixture
def library_parses(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the source label of every module parse the loader performs."""
    labels: list[str] = []
    original = loader_module.parse_program_seeded

    def counting(*args: object, **kwargs: object) -> object:
        labels.append(getattr(kwargs.get("source"), "label", ""))
        return original(*args, **kwargs)

    monkeypatch.setattr(loader_module, "parse_program_seeded", counting)
    return labels


def _write_module(root: Path, module_path: str, source: str) -> Path:
    path = root / (module_path.replace("/", os.sep) + ".agl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _stdlib_roots(root: Path) -> RootSet:
    """A root set that treats *root* as a host-selected standard library."""
    return RootSet(roots=frozenset({root}), stdlib_roots=frozenset({root}))


def _loose_roots(root: Path) -> RootSet:
    """A root set with no standard-library root at all."""
    return RootSet(roots=frozenset({root}))


def _load(roots: RootSet) -> LoadedModule:
    graph = load_graph(_ENTRY_SOURCE, entry_path=None, roots=roots, default_stdlib=False)
    return graph.modules[_LIB_ID]


def _library_parse_count(labels: list[str], path: Path) -> int:
    return sum(1 for label in labels if label == str(path))


# ---------------------------------------------------------------------------
# Loader integration
# ---------------------------------------------------------------------------


def test_standard_library_module_is_parsed_once_across_compilations(
    tmp_path: Path, library_parses: list[str]
) -> None:
    path = _write_module(tmp_path, "lib/a", _LIB_SOURCE)
    roots = _stdlib_roots(tmp_path)

    first = _load(roots)
    second = _load(roots)

    assert _library_parse_count(library_parses, path) == 1
    assert second.program is first.program


def test_an_ordinary_module_is_parsed_once_across_compilations(
    tmp_path: Path, library_parses: list[str]
) -> None:
    """Nothing about the standard library makes it uniquely cacheable.

    An ordinary module under no standard-library root is reused on exactly the
    same terms: for as long as its bytes are unchanged.
    """
    path = _write_module(tmp_path, "lib/a", _LIB_SOURCE)
    roots = _loose_roots(tmp_path)

    first = _load(roots)
    second = _load(roots)

    assert _library_parse_count(library_parses, path) == 1
    assert second.program is first.program


@pytest.mark.parametrize("build_roots", [_stdlib_roots, _loose_roots])
def test_a_rewrite_under_a_restored_timestamp_is_reparsed(
    tmp_path: Path, library_parses: list[str], build_roots: Callable[[Path], RootSet]
) -> None:
    """Stat metadata cannot separate these two sources, but the bytes can.

    A file rewritten in place, to the same length, under its original
    modification time presents an identical (mtime, size, inode) stamp. A cache
    that trusted metadata would serve the stale parse; the window is narrow but
    a process that both writes and compiles modules reaches it.
    """
    original = "def f() -> int = 1\n"
    replacement = "def f() -> int = 7\n"
    assert len(original) == len(replacement)
    path = _write_module(tmp_path, "lib/a", original)
    roots = build_roots(tmp_path)
    before = path.stat()

    first = _load(roots)
    path.write_text(replacement)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert path.stat().st_ino == before.st_ino

    second = _load(roots)

    assert _library_parse_count(library_parses, path) == 2
    assert second.program is not first.program
    assert "= 7" in second.source_text


def test_modified_standard_library_file_is_reparsed(
    tmp_path: Path, library_parses: list[str]
) -> None:
    path = _write_module(tmp_path, "lib/a", _LIB_SOURCE)
    roots = _stdlib_roots(tmp_path)

    _load(roots)
    path.write_text("def f() -> int = 2\ndef g() -> int = 3\n")
    reloaded = _load(roots)

    assert _library_parse_count(library_parses, path) == 2
    assert "def g()" in reloaded.source_text


def test_clearing_memory_preserves_the_loaded_library(tmp_path: Path) -> None:
    _write_module(tmp_path, "lib/a", _LIB_SOURCE)
    roots = _stdlib_roots(tmp_path)
    first = _load(roots)
    clear_parsed_module_cache()
    second = _load(roots)
    assert second.program == first.program
    assert second.source_text == _LIB_SOURCE


def test_cached_node_ids_stay_disjoint_from_a_compilation_entry(tmp_path: Path) -> None:
    _write_module(tmp_path, "lib/a", _LIB_SOURCE)
    roots = _stdlib_roots(tmp_path)

    _load(roots)
    graph = load_graph(
        "import lib/a\ndef big() -> int = 1\ndef bigger() -> int = 2\n",
        entry_path=None,
        roots=roots,
        default_stdlib=False,
    )

    entry_ids = all_node_ids(graph.modules[ENTRY_ID].program)
    library_ids = all_node_ids(graph.modules[_LIB_ID].program)
    assert entry_ids
    assert min(library_ids) >= RESERVED_NODE_ID_BASE
    assert max(entry_ids) < RESERVED_NODE_ID_BASE
    assert not entry_ids & library_ids


def test_prelude_injection_is_cached_separately(tmp_path: Path, library_parses: list[str]) -> None:
    """A module loaded with and without the prelude is two distinct results."""
    path = _write_module(tmp_path, "lib/a", _LIB_SOURCE)
    stdlib_source = Path(__file__).resolve().parents[1] / "packages" / "stdlib"
    roots = RootSet(
        roots=frozenset({tmp_path, stdlib_source}),
        stdlib_roots=frozenset({tmp_path, stdlib_source}),
    )

    without = load_graph(_ENTRY_SOURCE, entry_path=None, roots=roots, default_stdlib=False).modules[
        _LIB_ID
    ]
    with_prelude = load_graph(
        _ENTRY_SOURCE, entry_path=None, roots=roots, default_stdlib=True
    ).modules[_LIB_ID]

    assert _library_parse_count(library_parses, path) == 2
    assert len(with_prelude.imports) == len(without.imports) + 1


def test_missing_extern_companion_is_reported_after_a_cache_hit(tmp_path: Path) -> None:
    module_path = _write_module(tmp_path, "lib/a", "extern def f() -> int\n")
    companion = module_path.with_suffix(".py")
    companion.write_text("def f():\n    return 1\n")
    roots = _stdlib_roots(tmp_path)

    assert _load(roots).companion_path == companion
    companion.unlink()
    with pytest.raises(MissingExternCompanion):
        _load(roots)


def test_same_size_content_change_is_reparsed(tmp_path: Path, library_parses: list[str]) -> None:
    """A rewrite that keeps the byte count is still a different file.

    Size alone cannot separate these two sources, so this pins that the
    identity stamp is not size-only and no compilation is served the old parse.
    """
    original = "def f() -> int = 1\n"
    replacement = "def f() -> int = 7\n"
    assert len(original) == len(replacement)
    path = _write_module(tmp_path, "lib/a", original)
    roots = _stdlib_roots(tmp_path)

    first = _load(roots)
    path.write_text(replacement)
    second = _load(roots)

    assert _library_parse_count(library_parses, path) == 2
    assert first.source_text == original
    assert second.source_text == replacement


def _companion_transcript(root: Path, roots: RootSet) -> list[object]:
    """Run one extern program twice, rewriting its companion in between."""
    module_path = _write_module(root, "lib/a", "extern def f() -> int\n")
    companion = module_path.with_suffix(".py")
    companion.write_text("def f():\n    return 1\n")
    runtime = PipelineDriver()
    source = "import lib/a\nlet v = lib/a::f()"

    first = run_inline_command(runtime, source, roots=roots, default_stdlib=False)
    companion.write_text("def f():\n    return 9\n")
    second = run_inline_command(runtime, source, roots=roots, default_stdlib=False)

    assert first.ok and second.ok, (first.diagnostics, second.diagnostics)
    return [first.bindings["v"], second.bindings["v"]]


def test_a_rewritten_companion_behaves_the_same_cached_and_uncached(tmp_path: Path) -> None:
    """A companion's contents are never part of a parse, so a hit changes nothing.

    The cache re-verifies that a companion still exists but not what it holds,
    which is why this compares the two loader paths rather than asserting an
    outcome: whatever a rewritten companion does for an uncached module, a
    cache hit must do the same.
    """
    cached_root = tmp_path / "cached"
    loose_root = tmp_path / "loose"

    cached = _companion_transcript(cached_root, _stdlib_roots(cached_root))
    uncached = _companion_transcript(loose_root, _loose_roots(loose_root))

    assert cached == uncached


def test_module_with_tab_indentation_is_reparsed(tmp_path: Path, library_parses: list[str]) -> None:
    """Tab advisories reach the ambient collector, so such a module never caches."""
    path = _write_module(tmp_path, "lib/a", "def f() -> int =\n\t1\n")
    roots = _stdlib_roots(tmp_path)

    _load(roots)
    _load(roots)

    assert _library_parse_count(library_parses, path) == 2


# ---------------------------------------------------------------------------
# Whole-pipeline equivalence
# ---------------------------------------------------------------------------


def test_warm_cache_preserves_program_behavior(capsys: pytest.CaptureFixture[str]) -> None:
    runtime = PipelineDriver()
    source = 'print("hi")\nprint(2 + 3)'

    first = run_inline_command(runtime, source)
    first_output = capsys.readouterr().out
    second = run_inline_command(runtime, source)
    second_output = capsys.readouterr().out

    assert first.ok and second.ok
    assert first_output == second_output
    assert [d.message for d in first.warnings] == [d.message for d in second.warnings]


def test_warm_cache_preserves_diagnostics() -> None:
    first = prepare_inline_command("print(no_such_name)")
    second = prepare_inline_command("print(no_such_name)")

    assert first.diagnostics
    assert [d.message for d in first.diagnostics] == [d.message for d in second.diagnostics]
    assert [d.line for d in first.diagnostics] == [d.line for d in second.diagnostics]


# ---------------------------------------------------------------------------
# Cache unit behavior
# ---------------------------------------------------------------------------


def _stub_builder(module: LoadedModule, consumed: int, calls: list[int]) -> object:
    def build(start_id: int, source_text: str) -> tuple[LoadedModule, int]:
        calls.append(start_id)
        return module, start_id + consumed

    return build


def test_reopened_libraries_keep_distinct_declarations(tmp_path: Path) -> None:
    _write_module(tmp_path, "lib/a", _LIB_SOURCE)
    _write_module(tmp_path, "lib/b", _LIB_SOURCE)
    roots = _stdlib_roots(tmp_path)
    graph = load_graph(
        "import lib/a\nimport lib/b\n", roots=roots, entry_path=None, default_stdlib=False
    )
    first = all_node_ids(graph.modules[ModuleId.from_path("lib/a")].program)
    second = all_node_ids(graph.modules[ModuleId.from_path("lib/b")].program)
    assert first.isdisjoint(second)
    clear_parsed_module_cache()
    reopened = load_graph(
        "import lib/b\nimport lib/a\n", roots=roots, entry_path=None, default_stdlib=False
    )
    assert all_node_ids(reopened.modules[ModuleId.from_path("lib/a")].program) == first
    assert all_node_ids(reopened.modules[ModuleId.from_path("lib/b")].program) == second


def test_derivation_cache_evicts_least_recently_used_entries(tmp_path: Path) -> None:
    """A bounded derivation store drops its coldest entry rather than growing."""
    _write_module(tmp_path, "lib/a", _LIB_SOURCE)
    module = _load(_stdlib_roots(tmp_path))
    cache: ModuleDerivationCache[str, int] = ModuleDerivationCache(capacity=1)
    builds: list[str] = []

    def build(key: str) -> Callable[[], int]:
        def run() -> int:
            builds.append(key)
            return len(builds)

        return run

    cache.get_or_build(module, key="first", build=build("first"))
    cache.get_or_build(module, key="second", build=build("second"))
    cache.get_or_build(module, key="first", build=build("first"))

    assert builds == ["first", "second", "first"]


def test_parsed_cache_preserves_recent_modules_across_eviction(tmp_path: Path) -> None:
    for name in ("a", "b", "c"):
        _write_module(tmp_path, f"lib/{name}", _LIB_SOURCE)
    graph = load_graph(
        "import lib/a\nimport lib/b\nimport lib/c\n",
        roots=_stdlib_roots(tmp_path),
        entry_path=None,
        default_stdlib=False,
    )
    cache = ParsedModuleCache(capacity=2)

    def cached(name: str) -> LoadedModule:
        module_id = ModuleId.from_path(f"lib/{name}")
        module = graph.modules[module_id]

        def build(start_id: int, source_text: str) -> tuple[LoadedModule, int]:
            return replace(module), max(all_node_ids(module.program)) + 1

        return cache.get_or_build(
            module_id, tmp_path / f"lib/{name}.agl", default_stdlib=False, build=build
        )

    first = cached("a")
    second = cached("b")
    assert cached("a") is first
    cached("c")
    assert cached("a") is first
    reopened = cached("b")
    assert reopened is not second
    assert reopened.program == second.program


def test_infix_chain_module_keeps_one_resolved_program_across_compilations(
    tmp_path: Path, library_parses: list[str]
) -> None:
    """A library module whose bodies hold infix chains is resolved once.

    Infix-chain resolution rewrites a module's program, so a library module
    re-resolved per compilation would hand every later pass a fresh object and
    silently defeat the identity-keyed scope and type-check reuse guards.
    """
    path = _write_module(tmp_path, "lib/a", "def f(x: int) -> int = x + 1 + 2\n")
    roots = _stdlib_roots(tmp_path)

    first = _load(roots)
    second = _load(roots)

    assert _library_parse_count(library_parses, path) == 1
    assert second.program is first.program
