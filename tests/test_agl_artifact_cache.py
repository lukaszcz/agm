"""Repeated program executions observe current source and isolated host state."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agm.agl import artifact_cache, artifact_storage
from agm.agl.capabilities import HostCapabilities
from agm.agl.lower.program import lower_program
from agm.agl.matchcompile import compile_program_matches
from agm.agl.modules.ids import ModuleId
from agm.agl.modules.loader import build_repl_graph, parse_entry_module
from agm.agl.modules.parsed_module_cache import clear_parsed_module_cache
from agm.agl.pipeline import PipelineDriver
from agm.agl.runtime.codec import TextCodec
from agm.agl.scope.program import ResolvedProgram, resolve_program
from agm.agl.syntax.nodes import static_function_items, static_type_items
from agm.agl.typecheck.env import (
    AglTypeError,
    CheckedModule,
    CheckedModuleImage,
    EnvironmentFacts,
    TypeEnvironment,
)
from agm.agl.typecheck.program import _prepare_program, check_program
from tests._agl_helpers import agl_roots, run_inline_command
from tests.agl.ir_harness import (
    base_caps,
    make_file_graph_from_files,
    make_inline_graph_from_files,
    write_module_file,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("print([1, 2, 3].map(fn(n: int) -> int => n * 2)[2])", "6\n"),
        ("print(case 1 of | 1 => 2 | _ => 3)", "2\n"),
        ("var counter = 0\ncounter := counter + 1\nprint(counter)", "1\n"),
    ],
    ids=["library-method", "match", "fresh-mutable-state"],
)
def test_repeated_executions_produce_the_expected_output(
    source: str, expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = PipelineDriver()
    for _ in range(2):
        result = run_inline_command(runtime, source)
        assert result.ok, result.diagnostics
        assert capsys.readouterr().out == expected


@pytest.mark.parametrize(
    ("replacement", "expected"),
    [("def helper() -> int = 7\n", "7\n"), ('def helper() -> text = "new"\n', "new\n")],
    ids=["same-size-edit", "changed-return-type"],
)
def test_edited_imports_change_the_next_execution(
    tmp_path: Path, replacement: str, expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "helper.agl"
    path.write_text("def helper() -> int = 1\n")
    roots = agl_roots(tmp_path)
    runtime = PipelineDriver()
    source = "import helper::*\nprint(helper())\n"
    assert run_inline_command(runtime, source, roots=roots).ok
    assert capsys.readouterr().out == "1\n"

    path.write_text(replacement)
    result = run_inline_command(runtime, source, roots=roots)

    assert result.ok, result.diagnostics
    assert capsys.readouterr().out == expected


def test_field_default_survives_the_module_cache_disk_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A cached library module's constructor field default round-trips
    through ``LoweredModule.capture()``/``link_into()`` and an on-disk
    artifact restore.

    ``clear_retained_artifacts()`` drops only in-memory retention (disk
    persists), forcing the second compile to rehydrate ``lib``'s
    ``LoweredModule`` -- and with it, ``NominalDescriptor.field_defaults`` for
    its defaulted record -- from ``artifact_serialization`` rather than reuse
    the in-process object.
    """
    (tmp_path / "lib.agl").write_text("record Limits\n  retries: int = 3\n")
    roots = agl_roots(tmp_path)
    runtime = PipelineDriver()
    source = "import lib::*\nprint(Limits().retries)\n"

    artifact_cache.clear_retained_artifacts()
    assert run_inline_command(runtime, source, roots=roots).ok
    assert capsys.readouterr().out == "3\n"

    artifact_cache.clear_retained_artifacts()  # drop memory only; disk persists
    result = run_inline_command(runtime, source, roots=roots)

    assert result.ok, result.diagnostics
    assert capsys.readouterr().out == "3\n"


def test_a_warm_checked_module_cache_still_resolves_an_imported_var_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unchanged library module served from the warm cache still exports its var."""
    path = tmp_path / "store.agl"
    path.write_text("var total: int = 0\n")
    roots = agl_roots(tmp_path)
    runtime = PipelineDriver()
    source = "import store::*\ntotal := total + 5\nprint(total)\n"
    for _ in range(2):
        result = run_inline_command(runtime, source, roots=roots)
        assert result.ok, result.diagnostics
        assert capsys.readouterr().out == "5\n"


def test_a_warm_checked_module_cache_still_resolves_an_unannotated_imported_var_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unchanged library module served from the warm cache still exports its
    unannotated var, with its published binding type read straight off the
    cached module instead of being re-inferred."""
    path = tmp_path / "store.agl"
    path.write_text("var total = 0\n")
    roots = agl_roots(tmp_path)
    runtime = PipelineDriver()
    source = "import store::*\ntotal := total + 5\nprint(total)\n"
    for _ in range(2):
        result = run_inline_command(runtime, source, roots=roots)
        assert result.ok, result.diagnostics
        assert capsys.readouterr().out == "5\n"


def test_exception_downcast_and_is_stay_stable_across_a_warm_artifact_cache_reuse(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A descendant declared only in the entry module still conforms, via
    `as?`, `is`, and a `catch` of the library's base type, after the library
    module's checked/lowered artifacts are served from the warm cache."""
    lib_path = tmp_path / "lib.agl"
    lib_path.write_text(
        "exception Problem extends Exception\n"
        "  code: int\n"
        "exception Detailed extends Problem\n"
        "  detail: text\n"
        "def detail-of(e: Problem) -> Option[Detailed] = e as? Detailed\n"
        "def is-detailed(e: Problem) -> bool = e is Detailed\n"
        "def guarded(f: () -> int) -> int =\n"
        "  try f() catch Problem as p => p.code\n"
    )
    roots = agl_roots(tmp_path)
    runtime = PipelineDriver()
    source = """\
import lib::*
exception VeryDetailed extends Detailed
  hint: text
let v = VeryDetailed(message = "m", code = 1, detail = "d", hint = "h")
case detail-of(v) of
  | Some(value) => print(value.detail)
  | None => print("none")
print(is-detailed(v))
print(guarded(fn() => raise VeryDetailed(message = "m", code = 2, detail = "d", hint = "h")))
"""
    for _ in range(2):
        result = run_inline_command(runtime, source, roots=roots)
        assert result.ok, result.diagnostics
        assert capsys.readouterr().out == "d\ntrue\n2\n"


def test_editing_an_unannotated_exported_binding_type_invalidates_the_warm_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A source edit that changes an unannotated exported binding's inferred
    type invalidates the warm cache: the importer never sees a stale type."""
    path = tmp_path / "store.agl"
    path.write_text("let value = 1\n")
    roots = agl_roots(tmp_path)
    runtime = PipelineDriver()
    source = "import store::*\nprint(value)\n"
    result = run_inline_command(runtime, source, roots=roots)
    assert result.ok, result.diagnostics
    assert capsys.readouterr().out == "1\n"

    path.write_text('let value = "changed"\n')
    result = run_inline_command(runtime, source, roots=roots)
    assert result.ok, result.diagnostics
    assert capsys.readouterr().out == "changed\n"


def test_warm_cached_importer_reflects_a_changed_unannotated_transitive_binding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A cached library importer's own source never changes, but its
    transitive dependency's unannotated binding does: the importer is
    rechecked (not served stale from cache) and observes the new type/value."""
    store_path = tmp_path / "store.agl"
    mid_path = tmp_path / "mid.agl"
    mid_path.write_text("import store\ndef get() = store::value\n")
    roots = agl_roots(tmp_path)
    runtime = PipelineDriver()
    source = "import mid::*\nprint(get())\n"

    store_path.write_text("let value = 1\n")
    result = run_inline_command(runtime, source, roots=roots)
    assert result.ok, result.diagnostics
    assert capsys.readouterr().out == "1\n"

    store_path.write_text('let value = "changed"\n')
    result = run_inline_command(runtime, source, roots=roots)
    assert result.ok, result.diagnostics
    assert capsys.readouterr().out == "changed\n"


def test_editing_a_folded_constant_invalidates_the_warm_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A constant a module folds into another of its bindings is part of that
    module's own source, so editing it re-lowers the module rather than
    serving the previously folded text."""
    path = tmp_path / "store.agl"
    path.write_text('let part = "one"\nlet whole = "[%{part}]"\n')
    roots = agl_roots(tmp_path)
    runtime = PipelineDriver()
    source = "import store::*\nprint(whole)\n"
    result = run_inline_command(runtime, source, roots=roots)
    assert result.ok, result.diagnostics
    assert capsys.readouterr().out == "[one]\n"

    path.write_text('let part = "two"\nlet whole = "[%{part}]"\n')
    result = run_inline_command(runtime, source, roots=roots)
    assert result.ok, result.diagnostics
    assert capsys.readouterr().out == "[two]\n"


def test_invalid_import_edit_is_rejected_and_can_be_repaired(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "helper.agl"
    roots = agl_roots(tmp_path)
    runtime = PipelineDriver()
    source = "import helper::*\nprint(helper())\n"
    path.write_text("def helper() -> int = 1\n")
    assert run_inline_command(runtime, source, roots=roots).ok
    assert capsys.readouterr().out == "1\n"

    path.write_text('def helper() -> int = "bad"\n')
    failed = run_inline_command(runtime, source, roots=roots)
    assert not failed.ok
    assert failed.diagnostics
    assert capsys.readouterr().out == ""

    path.write_text("def helper() -> int = 9\n")
    assert run_inline_command(runtime, source, roots=roots).ok
    assert capsys.readouterr().out == "9\n"


def test_same_named_imports_are_isolated_between_roots(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = PipelineDriver()
    for name, value in (("first", 1), ("second", 7), ("first", 1)):
        root = tmp_path / name
        root.mkdir(exist_ok=True)
        (root / "helper.agl").write_text(f"def helper() -> int = {value}\n")
        result = run_inline_command(
            runtime, "import helper::*\nprint(helper())\n", roots=agl_roots(root)
        )
        assert result.ok, result.diagnostics
        assert capsys.readouterr().out == f"{value}\n"


def test_discarding_compilation_state_preserves_program_behavior(
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = "print([1, 2, 3].map(fn(n: int) -> int => n * 2)[2])"
    runtime = PipelineDriver()
    assert run_inline_command(runtime, source).ok
    assert capsys.readouterr().out == "6\n"

    artifact_cache.clear_retained_artifacts()

    assert run_inline_command(runtime, source).ok
    assert capsys.readouterr().out == "6\n"


def test_non_entry_cycle_member_is_retained_against_the_entry_source(tmp_path: Path) -> None:
    graph = make_file_graph_from_files(
        tmp_path,
        {"entry": "import helper\n()", "helper": "()"},
        default_stdlib=False,
    )
    entry = graph.modules[graph.entry_id]
    graph.modules[graph.entry_id] = replace(entry, path=tmp_path / "entry.agl")
    helper_id = next(module_id for module_id in graph.modules if module_id != graph.entry_id)
    graph.adjacency[helper_id] = (graph.entry_id,)

    retained = artifact_cache.retained_module_sources(graph)

    assert tuple(module.module_id for module in retained[helper_id]) == (
        graph.entry_id,
        helper_id,
    )
    assert graph.entry_id not in retained


def test_config_attribute_program_configs_round_trip_through_the_resolved_cache(
    tmp_path: Path,
) -> None:
    """A program def's @config entries survive the disk-backed resolved-module cache."""
    graph = make_file_graph_from_files(
        tmp_path,
        {
            "entry": "import helper\n\nprogram def main() -> unit = ()\n",
            "helper": (
                "import std/config\n\n"
                "@config(config::trace = true)\n"
                "program def build() -> unit = ()\n"
            ),
        },
    )
    resolved_program = resolve_program(graph)
    retainable = artifact_cache.retained_module_sources(graph)
    artifact_cache.retain_resolved_modules(retainable, resolved_program.modules)
    artifact_cache.clear_retained_artifacts()

    restored = artifact_cache.retained_resolved_modules(retainable)

    helper_id = next(mid for mid in graph.modules if mid != graph.entry_id)
    original = resolved_program.modules[helper_id].resolved.attributes.program_configs
    round_tripped = restored[helper_id].resolved.attributes.program_configs
    assert original
    assert round_tripped == original


def test_config_attribute_program_config_targets_round_trip_through_the_checked_cache(
    tmp_path: Path,
) -> None:
    """A program def's resolved '@config' targets survive the disk-backed checked-module cache."""
    graph = make_file_graph_from_files(
        tmp_path,
        {
            "entry": "import helper\n\nprogram def main() -> unit = ()\n",
            "helper": (
                "import std/config\n\n"
                "@config(config::trace = true)\n"
                "program def build() -> unit = ()\n"
            ),
        },
    )
    resolved_program = resolve_program(graph)
    caps = base_caps()
    retainable = artifact_cache.retained_module_sources(graph)
    checked = check_program(resolved_program, caps)
    artifact_cache.retain_checked_modules(retainable, caps, checked.modules)
    artifact_cache.clear_retained_artifacts()

    restored = artifact_cache.retained_checked_modules(retainable, caps)

    helper_id = next(mid for mid in graph.modules if mid != graph.entry_id)
    original = checked.modules[helper_id].program_config_targets
    round_tripped = restored[helper_id]
    assert original
    assert isinstance(round_tripped, CheckedModuleImage)
    assert round_tripped.program_config_targets == original


def test_a_params_flip_invalidates_a_warm_checked_module_cache(tmp_path: Path) -> None:
    """A cross-module ``@config`` target's ``@param``-ness is rechecked, not cached stale."""
    entry_source = "import helper\n\n@config(helper::value = 2)\nprogram def main() -> unit = ()\n"

    def _round_trip(helper_source: str) -> None:
        graph = make_file_graph_from_files(
            tmp_path, {"entry": entry_source, "helper": helper_source}
        )
        resolved_program = resolve_program(graph)
        retainable = artifact_cache.retained_module_sources(graph)
        cached = artifact_cache.retained_checked_modules(retainable, base_caps())
        checked = check_program(resolved_program, base_caps(), cached_checked_modules=cached)
        artifact_cache.retain_checked_modules(retainable, base_caps(), checked.modules)

    _round_trip("@param let value: int = 1\n")

    with pytest.raises(AglTypeError):
        _round_trip("let value: int = 1\n")

    _round_trip("@param let value: int = 1\n")


def _persisted_checked_entry_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str, imports: tuple[str, ...]
) -> int:
    """Compile a library importing *imports* stdlib modules and size its ``checked`` entry."""
    root = tmp_path / scenario / "root"
    root.mkdir(parents=True)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / scenario / "cache"))
    library_source = "".join(f"import std/{name}::*\n" for name in imports) + (
        'def greet() -> text = "hi"\n'
    )
    write_module_file(root, "library", library_source)
    clear_parsed_module_cache()
    artifact_cache.clear_retained_artifacts()
    parsed = parse_entry_module("import library::*\ngreet()", entry_path=None, inline_command=True)
    graph, _next_id, _new_modules = build_repl_graph(
        parsed.program,
        parsed.next_id,
        path=None,
        cached={},
        roots=agl_roots(root, include_stdlib=True),
        default_stdlib=False,
        source_text="import library::*\ngreet()",
    )
    caps = base_caps()
    resolved = resolve_program(graph)
    check_program(resolved, caps)
    library_id = next(mid for mid in graph.modules if mid.path_str() == "library")
    retainable = artifact_cache.retained_module_sources(graph)
    key = artifact_cache._disk_key(
        retainable[library_id], (artifact_cache.capability_signature(caps),)
    )
    path, _identity = artifact_storage.artifact_entry(key, "checked")
    return path.stat().st_size


def test_a_persisted_checked_entry_stays_small_regardless_of_import_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The disk artifact holds only a module's own facts, not the program's type environment.

    Before rehydration, a persisted ``checked`` entry embedded a copy of the
    whole-program type environment, so its size scaled with everything the
    module could see -- not with what it declares. ``assert one < budget`` is
    the assertion with real teeth: importing even one stdlib module used to
    pull that module's whole-program environment onto disk, well past this
    budget. The other two assertions only tighten the bound: importing twelve
    stdlib modules must not cost dramatically more to persist than one.
    """
    twelve_modules = (
        "text",
        "array",
        "dict",
        "math",
        "option",
        "pair",
        "either",
        "result",
        "time",
        "random",
        "json",
        "toml",
    )
    one = _persisted_checked_entry_size(tmp_path, monkeypatch, "one", ("text",))
    twelve = _persisted_checked_entry_size(tmp_path, monkeypatch, "twelve", twelve_modules)

    # This fixture is a single-member dependency group (one library, one
    # entry importing it): a multi-member group's shared entry can legitimately
    # persist larger, since all members' facts share one disk entry.
    budget = 32 * 1024
    assert one < budget
    assert twelve < budget
    assert twelve < one * 4


def test_rehydration_onto_a_cached_preparation_matches_a_cache_free_compile(
    tmp_path: Path,
) -> None:
    """Rehydrating onto an environment prepared with ``cached_checked_modules`` still
    reproduces every declaration query the spec enumerates for a checked module.

    ``cached_checked_modules`` short-circuits per-module body checking for the
    library, so its own preparation (the method-header registration loop) must
    read the unannotated method's and function's inferred signature -- recorded
    only via candidate inference, never journaled -- identically whether the
    library is a live object or a disk-served image. This reproduces a real
    regression where that loop read ``.resolved`` off the image and crashed
    with an ``AttributeError``, because a ``CheckedModuleImage`` carries no
    ``resolved``. Beyond function signatures and binding types, this also
    checks the whole-module queries (own-facts journal, closed interface,
    named-type/template resolution, enum forms, generic types) that a
    body-check-skipping preparation could equally leave stale.
    """
    graph = make_inline_graph_from_files(
        tmp_path,
        {
            "entry": "import lib::*\nincrement(Box(value = 1).double())",
            "lib": (
                "record Box(value: int)\n"
                "def Box::double(self) = self.value * 2\n"
                "def increment(x: int) = x + 1\n"
                "def triple(x: int) = x * 3\n"
                "record Pair[A, B](first: A, second: B)\n"
                "enum Shape\n"
                "  | circle(radius: int)\n"
                "  | square(side: int)\n"
            ),
        },
        default_stdlib=False,
    )
    caps = base_caps()
    resolved = resolve_program(graph)
    checked = check_program(resolved, caps)
    lib_id = next(mid for mid in resolved.modules if mid.path_str() == "lib")
    cm = checked.modules[lib_id]
    image = cm.image()

    prepared = _prepare_program(resolved, caps, cached_checked_modules={lib_id: image})
    rehydrated = image.rehydrate(resolved.modules[lib_id], prepared.module_envs[lib_id])

    for item in static_function_items(cm.resolved.program.body.items):
        assert rehydrated.type_env.get_function_signature_by_node_id(
            item.node_id
        ) == cm.type_env.get_function_signature_by_node_id(item.node_id)
        for param in item.params:
            assert rehydrated.type_env.get_binding_type(
                param.node_id
            ) == cm.type_env.get_binding_type(param.node_id)

    # own_facts / interface: whole-environment summaries a stale header loop
    # could leave desynchronized from the image they were meant to reproduce.
    assert rehydrated.type_env.own_facts() == image.environment_facts
    assert rehydrated.interface == image.interface

    for item in static_type_items(cm.resolved.program.body.items):
        scope_path = tuple(segment.name for segment in item.scope_path)
        with rehydrated.type_env.type_scope(scope_path):
            re_named = rehydrated.type_env.resolve_named_type(item.name)
        with cm.type_env.type_scope(scope_path):
            cm_named = cm.type_env.resolve_named_type(item.name)
        assert re_named == cm_named
        assert rehydrated.type_env.source_type_template_qname(
            lib_id, item.name, scope_path=scope_path
        ) == cm.type_env.source_type_template_qname(lib_id, item.name, scope_path=scope_path)

    # enum forms / generic types: registries populated by the same header loop.
    assert rehydrated.type_env.enum_owner_forms() == cm.type_env.enum_owner_forms()
    assert rehydrated.type_env.all_generic_types() == cm.type_env.all_generic_types()


_CANDIDATE_CALLEE_SRC = "def factorial(n: int) = if n <= 1 => 1 else => n * factorial(n - 1)\n"
_CANDIDATE_CONSUMER_SRC = (
    "import lib_candidate_callee::*\ndef increment-factorial(n: int) -> int = factorial(n) + 1\n"
)
_UNRELATED_SRC = "def unrelated-value() -> int = 5\n"
_ENTRY_SRC = (
    "import lib_candidate_consumer::*\n"
    "import lib_unrelated::*\n"
    "def summarize() -> int = increment-factorial(3) + unrelated-value()\n"
)


def _method_registry_content(registry: object) -> list[tuple[str, ...]]:
    """Flatten a ``TypeTable`` method registry into an order-free comparable form."""
    assert isinstance(registry, dict)
    return sorted(
        (repr(owner), name, repr(declaration_key), repr(method))
        for owner, names in registry.items()
        for name, candidates in names.items()
        for declaration_key, method in candidates.items()
    )


def _normalize_env_value(value: object) -> object:
    """Make a ``TypeEnvironment`` attribute value comparable across two preparations.

    A ``TypeTable`` has no ``__eq__`` (identity equality), yet the two
    preparations under comparison legitimately build distinct instances with
    the same content, so it is compared by content instead: its declaration
    entries, the name index that decides which of them a name path resolves
    to, and its method registries -- declaration data of their own, reachable
    through no whole-table accessor. Bound methods/functions stored as
    attribute values are never meaningfully comparable across instances, so
    they are treated as always equal.
    """
    from agm.agl.semantics.type_table import TypeTable

    if isinstance(value, TypeTable):
        return (
            "TypeTable",
            sorted(repr(entry) for entry in value.entries()),
            sorted(repr(item) for item in vars(value)["_name_index"].items()),
            _method_registry_content(vars(value)["_methods"]),
            _method_registry_content(vars(value)["_builtin_methods"]),
        )
    if callable(value) and not isinstance(value, type):
        return "<callable>"
    return value


def test_prepared_environments_match_cache_free_across_every_module(tmp_path: Path) -> None:
    """A cached preparation must reproduce a cache-free one for EVERY module's
    ``TypeEnvironment``, not just the cached module's own declarations.

    Regression test: ``_prepare_module_environment``'s bulk seeding step used
    to seed the whole ``program_func_sig_table`` -- including candidate
    (unannotated-function) records restored from ``cached_checked_modules``
    -- into every module environment, name-keyed as well as node-id-keyed. A
    freshly inferred candidate is published far more narrowly (by
    ``function_inference._register_signature``): by name only into its own
    declaring module, by node id only into its own import SCC and later ones.
    So supplying a cache-served image for ``lib_candidate_callee`` (whose
    ``factorial`` is unannotated, hence a candidate) used to leak
    ``factorial`` into ``lib_unrelated``'s environment, even though
    ``lib_unrelated`` neither imports nor is imported by it.

    Table contents are compared by reading ``TypeEnvironment`` instance
    attributes directly (via ``vars()``), because several of the tables this
    bug touches (``_binding_types``, ``_function_signatures_by_node_id``) have
    no public enumeration accessor -- only single-key lookups
    (``get_binding_type``, ``get_function_signature_by_node_id``) -- and
    ``all_function_signatures()`` exposes only the root-scope name-keyed
    table, not the other two.
    ``_import_env``/``_scope_nodes`` are excluded: both preparations share the
    identical input objects for them by construction, so they carry no
    information about this bug. Every module is compared, the entry included.
    """
    graph = make_inline_graph_from_files(
        tmp_path,
        {
            "entry": _ENTRY_SRC,
            "lib_candidate_callee": _CANDIDATE_CALLEE_SRC,
            "lib_candidate_consumer": _CANDIDATE_CONSUMER_SRC,
            "lib_unrelated": _UNRELATED_SRC,
        },
        default_stdlib=False,
    )
    caps = base_caps()
    resolved = resolve_program(graph)
    checked = check_program(resolved, caps)
    callee_id = next(mid for mid in resolved.modules if mid.path_str() == "lib_candidate_callee")
    image = checked.modules[callee_id].image()

    fresh = _prepare_program(resolved, caps)
    cached = _prepare_program(resolved, caps, cached_checked_modules={callee_id: image})

    ignored_attrs = {"_import_env", "_scope_nodes"}
    diffs: list[str] = []
    for mid, fresh_env in fresh.module_envs.items():
        cached_env = cached.module_envs[mid]
        for name, fresh_value in vars(fresh_env).items():
            if name in ignored_attrs:
                continue
            cached_value = vars(cached_env).get(name, "<missing>")
            if _normalize_env_value(fresh_value) != _normalize_env_value(cached_value):
                diffs.append(f"{mid.path_str()}.{name}")
    assert not diffs, f"cached vs cache-free preparation diverged: {diffs}"


def test_check_program_lowers_identically_after_rehydrating_disk_checked_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``check_program`` itself, serving and rehydrating ``checked`` images off
    disk, lowers to the same program as a from-scratch compile.

    Image/rehydrate parity is proven elsewhere over a hand-reassembled
    ``CheckedProgram``; this drives the production path instead: one compile
    warms the disk cache, ``clear_retained_artifacts()`` drops only the
    in-memory retention, and the second compile must serve images from disk --
    which a spy on ``retained_checked_modules`` asserts actually happened, so
    an accidentally all-fresh second compile cannot pass.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    modules = {
        "entry": (
            "import lib_a::*\nimport lib_b::*\ndef summarize() -> int = double(3) + quadruple(2)\n"
        ),
        "lib_a": "def double(x: int) -> int = x * 2\n",
        "lib_b": "import lib_a::*\ndef quadruple(x: int) -> int = double(double(x))\n",
    }

    def _resolved() -> ResolvedProgram:
        clear_parsed_module_cache()
        graph = make_inline_graph_from_files(tmp_path, modules, default_stdlib=False)
        return resolve_program(graph)

    caps = base_caps()
    artifact_cache.clear_retained_artifacts()
    warm_checked = check_program(_resolved(), caps)  # populates the disk-backed "checked" entries
    original_executable = lower_program(compile_program_matches(warm_checked).compiled)

    artifact_cache.clear_retained_artifacts()  # drop memory only; disk persists

    from agm.agl.typecheck import program as program_module

    original_served = program_module.retained_checked_modules
    served_images: list[ModuleId] = []

    def spied(
        retainable: artifact_cache.RetainedSources, capabilities: HostCapabilities
    ) -> dict[ModuleId, CheckedModule | CheckedModuleImage]:
        served = original_served(retainable, capabilities)
        served_images.extend(
            mid for mid, module in served.items() if isinstance(module, CheckedModuleImage)
        )
        return served

    monkeypatch.setattr(program_module, "retained_checked_modules", spied)

    rehydrated_checked = check_program(_resolved(), caps)
    rehydrated_executable = lower_program(compile_program_matches(rehydrated_checked).compiled)

    assert served_images, "expected check_program to serve at least one disk-rehydrated image"
    assert rehydrated_executable == original_executable


def test_a_corrupted_binding_replay_fails_a_rehydrated_module_via_env_assert_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``TypeEnvironment.assert_closed()`` catches a broken-journal binding leak.

    A rehydrated module is never added to ``check_program``'s
    ``reused_modules``, so ``assert_checked_module_closed`` validates it like a
    freshly checked one -- specifically, the ``module.type_env.assert_closed()``
    half of that check, since ``set_binding_type`` mutates the type
    environment, not any of the ``CheckedModule`` fields
    ``assert_checked_output_closed`` inspects. Corrupting
    ``TypeEnvironment.replay`` to leak a stray inference variable simulates a
    broken own-facts journal and must make this assertion fire on the next
    compilation that rehydrates the disk-served image (see the sibling test
    below for the ``assert_checked_output_closed`` half, which this corruption
    does not exercise).
    """
    from agm.agl.semantics.types import InferenceVarType

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    library_source = "def value() -> int = 1\n"

    def _resolved() -> ResolvedProgram:
        clear_parsed_module_cache()
        graph = make_inline_graph_from_files(
            tmp_path,
            {"entry": "import lib::*\nvalue()", "lib": library_source},
            default_stdlib=False,
        )
        return resolve_program(graph)

    caps = base_caps()
    artifact_cache.clear_retained_artifacts()
    check_program(_resolved(), caps)  # populates the disk-backed "checked" entry

    artifact_cache.clear_retained_artifacts()  # drop memory only; disk persists

    original_replay = TypeEnvironment.replay

    def broken_replay(self: TypeEnvironment, facts: EnvironmentFacts) -> None:
        original_replay(self, facts)
        self.set_binding_type(node_id=-1, typ=InferenceVarType())

    monkeypatch.setattr(TypeEnvironment, "replay", broken_replay)

    with pytest.raises(AssertionError):
        check_program(_resolved(), caps)


def test_a_corrupted_node_types_entry_fails_a_rehydrated_module_via_output_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``assert_checked_output_closed`` catches a corrupted ``node_types`` entry.

    ``node_types`` lives on ``CheckedModuleImage``/``CheckedModule`` directly,
    outside the type environment ``TypeEnvironment.assert_closed()`` walks --
    the sibling test above corrupts a binding instead, which that other check
    alone catches. Wrapping ``retained_checked_modules`` to inject a stray
    inference variable into a served image's ``node_types`` proves the two
    checks ``_assert_checked_module_closed`` runs cover disjoint data: this
    corruption is invisible to ``env.assert_closed()`` and is caught only by
    ``assert_checked_output_closed``.
    """
    from agm.agl.semantics.types import InferenceVarType
    from agm.agl.typecheck import program as program_module

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    library_source = "def value() -> int = 1\n"

    def _resolved() -> ResolvedProgram:
        clear_parsed_module_cache()
        graph = make_inline_graph_from_files(
            tmp_path,
            {"entry": "import lib::*\nvalue()", "lib": library_source},
            default_stdlib=False,
        )
        return resolve_program(graph)

    caps = base_caps()
    artifact_cache.clear_retained_artifacts()
    check_program(_resolved(), caps)  # populates the disk-backed "checked" entry

    artifact_cache.clear_retained_artifacts()  # drop memory only; disk persists

    original = program_module.retained_checked_modules

    def corrupted(
        retainable: artifact_cache.RetainedSources, capabilities: HostCapabilities
    ) -> dict[ModuleId, CheckedModule | CheckedModuleImage]:
        served = original(retainable, capabilities)
        return {
            mid: (
                replace(module, node_types={**module.node_types, -1: InferenceVarType()})
                if isinstance(module, CheckedModuleImage)
                else module
            )
            for mid, module in served.items()
        }

    monkeypatch.setattr(program_module, "retained_checked_modules", corrupted)

    with pytest.raises(AssertionError):
        check_program(_resolved(), caps)


def test_an_in_memory_reused_module_skips_closure_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module reused verbatim from memory is trusted, not re-validated.

    It is the very object an earlier ``check_program`` call already sealed and
    validated; re-walking it on every subsequent compilation that imports it
    would be pure waste. Only a rehydrated or freshly checked module reaches
    ``_assert_checked_module_closed``.
    """
    calls: list[ModuleId] = []
    from agm.agl.typecheck import program as program_module

    original = program_module._assert_checked_module_closed

    def spied_assert_checked_module_closed(module: CheckedModule) -> None:
        calls.append(module.module_id)
        original(module)

    monkeypatch.setattr(
        program_module, "_assert_checked_module_closed", spied_assert_checked_module_closed
    )

    graph = make_inline_graph_from_files(
        tmp_path,
        {"entry": "import lib::*\nvalue()", "lib": "def value() -> int = 1\n"},
        default_stdlib=False,
    )
    caps = base_caps()
    resolved = resolve_program(graph)
    checked = check_program(resolved, caps)
    lib_id = next(mid for mid in resolved.modules if mid.path_str() == "lib")
    assert lib_id in calls
    calls.clear()

    # Same `resolved` object, same in-memory `CheckedModule`: the identity gate
    # in `check_program`'s reuse filter accepts it without rechecking its body.
    second = check_program(resolved, caps)

    assert lib_id not in calls
    # Trusted, not just present: reuse must be the identical object, not a
    # coincidentally-equal recheck that skipped validation for some other reason.
    assert second.modules[lib_id] is checked.modules[lib_id]


def test_the_image_is_bounded_and_evicts_the_least_recently_used() -> None:
    """Retention is capped, so a long-lived process cannot grow without limit.

    The store is exercised directly: filling the production cap through real
    compilations would mean building hundreds of distinct standard libraries.
    """
    store: artifact_cache._ArtifactStore[str] = artifact_cache._ArtifactStore(capacity=2)
    sources: artifact_cache.Sources = ()
    store.put(("a",), sources, "first")
    store.put(("b",), sources, "second")
    assert store.get(("a",), sources) == "first"

    store.put(("c",), sources, "third")

    assert store.get(("b",), sources) is None
    assert store.get(("a",), sources) == "first"
    assert store.get(("c",), sources) == "third"


def test_custom_response_formats_do_not_leak_between_hosts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class TaggedCodec(TextCodec):
        @property
        def name(self) -> str:
            return "tagged"

    extended = PipelineDriver(agent_dispatcher=lambda request: "answer")
    extended.register_codec(TaggedCodec())
    source = 'let answer: text = ask("prompt", format = "tagged")\nprint(answer)'
    assert run_inline_command(extended, source).ok
    assert capsys.readouterr().out == "answer\n"

    ordinary = PipelineDriver(
        agent_dispatcher=lambda _: pytest.fail("unsupported format dispatched")
    )
    rejected = run_inline_command(ordinary, source)
    assert not rejected.ok
    assert rejected.diagnostics or rejected.error is not None
    assert capsys.readouterr().out == ""

    assert run_inline_command(extended, source).ok
    assert capsys.readouterr().out == "answer\n"


_HEADER_LIB_SRC = (
    "type Count = int\n"
    "record Box(value: Count)\n"
    "record Pair[A, B](first: A, second: B)\n"
    "enum Shape\n"
    "  | circle(radius: int)\n"
    "  | square(side: int)\n"
    "exception Broken(reason: text)\n"
    "def Box::doubled(self) -> int = self.value * 2\n"
    "def Shape::sides(self) -> int = case self of | circle => 0 | square => 4\n"
    "def int::tripled(self) -> int = self * 3\n"
    "def widen(n: Count) -> int = n\n"
)
_HEADER_ENTRY_SRC = (
    "import lib_headers::*\n"
    "Box(value = 1).doubled() + circle(radius = 2).sides() + 5.tripled() + widen(1)\n"
)


def test_replayed_module_headers_reproduce_a_from_source_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Serving a module's headers from the memo must leave every environment
    exactly as preparing them from source would.

    Header preparation resolves each declaration's body and each declared
    function header, writing name-keyed types, generic templates, alias
    targets and frozen templates, constructor signatures and field kinds, and
    function signatures -- and, into the program-wide shared table,
    declaration identities and method headers. The memo replays a journal of
    those writes instead, so this drives a library declaring one of each and
    compares the WHOLE of every prepared ``TypeEnvironment`` (the entry
    module, which is never memoized, included) against a preparation that
    never consulted the memo. A spy on the lookup rules out a vacuous pass in
    which the second preparation silently re-prepared everything from source.
    """
    graph = make_inline_graph_from_files(
        tmp_path,
        {"entry": _HEADER_ENTRY_SRC, "lib_headers": _HEADER_LIB_SRC},
        default_stdlib=False,
    )
    caps = base_caps()
    resolved = resolve_program(graph)
    retainable = artifact_cache.retained_module_sources(resolved.graph)
    lib_id = next(mid for mid in resolved.modules if mid.path_str() == "lib_headers")

    from_source = _prepare_program(resolved, caps)
    _prepare_program(resolved, caps, retainable=retainable)

    served: list[dict[ModuleId, EnvironmentFacts]] = []
    lookup = artifact_cache.retained_module_headers
    monkeypatch.setattr(
        "agm.agl.typecheck.program.retained_module_headers",
        lambda *args: served.append(lookup(*args)) or served[-1],
    )
    replayed = _prepare_program(resolved, caps, retainable=retainable)

    assert served == [{lib_id: served[0][lib_id]}], "the library's headers were not served"
    assert served[0][lib_id].entries, "an empty journal would make the comparison vacuous"

    diffs = [
        f"{mid.path_str()}.{name}"
        for mid, source_env in from_source.module_envs.items()
        for name, value in vars(source_env).items()
        if name not in {"_import_env", "_scope_nodes"}
        and _normalize_env_value(value)
        != _normalize_env_value(vars(replayed.module_envs[mid]).get(name, "<missing>"))
    ]
    assert not diffs, f"replayed vs from-source header preparation diverged: {diffs}"
