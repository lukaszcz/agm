"""Real-module-graph test helpers for unit-level scope and typecheck tests.

Production never calls the per-module ``resolve_module``/``check_module``
entry points directly: it always builds a real
:class:`~agm.agl.modules.loader.ModuleGraph` (entry plus the standard
library) and runs :func:`~agm.agl.scope.program.resolve_program` /
:func:`~agm.agl.typecheck.program.check_program` over it. Calling the
per-module functions instead skips the whole-program pre-passes and the
entry's real import environment, so it can hide defects that only surface
under the configuration production actually runs.

:func:`resolve_entry` and :func:`resolve_and_check_entry` build a real
``ModuleGraph`` for a single entry source (using
:func:`~agm.agl.modules.loader.build_repl_graph`, the same seam the REPL
uses) and run the real program-level passes, returning the entry module's
``ModuleResolution``/``CheckedModule`` — the same shapes the old per-module
wrappers returned, so a call site conversion changes only the call.

The standard library (``std/core.agl``) is parsed once per process and
reused across calls via ``build_repl_graph``'s ``cached`` parameter, since
re-parsing it on every call would materially slow the suite (~17ms of the
~18ms a full load costs is spent parsing ``std/core.agl``). *default_stdlib*
therefore defaults to ``True``: that is the configuration production runs
(a bare ``print``/``exec``/``ask`` call resolves through its real
``std/core`` declaration, not through an undeclared-name fallback), and a
test should only opt out of it when being stdlib-free is itself the point
being tested — it declares its own ``builtin`` name (a bare builtin name may
be declared only once per program, and ``std/core`` already declares every
built-in name), or it asserts that some name is genuinely undefined with
nothing else in scope. It is never a substitute for fixing a test that reads
``program.body.items`` positionally — see below.

Contract for the returned ``program.body.items``: the helper's job is to
resolve *source* as a real program and hand back the entry module's
resolution, whose items are the ones the test wrote. Resolution and
type-checking always run with the injected ``std/core`` import present (that
is the real graph production builds), but the ``ModuleResolution`` returned
to the caller has that one synthetic ``ImportDecl`` removed from
``program.body.items`` — matched by the exact node id the helper itself
assigned it when calling ``build_repl_graph``, never by position or by "is
an ``ImportDecl``", so a test's own ``import`` declarations are untouched
and ``items[0]`` is always the test's own first statement. This is safe
because nothing else is re-keyed: every side table on ``ModuleResolution``
(and every table on ``CheckedModule``) is keyed by ``node_id``, not by
position in ``items``, so dropping the item from the list does not change
what any lookup returns.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from agm.agl.capabilities import HostCapabilities
from agm.agl.modules.ids import STD_CORE_ID, ModuleId
from agm.agl.modules.loader import LoadedModule, ModuleGraph, build_repl_graph
from agm.agl.modules.roots import RootSet
from agm.agl.parser.parser import parse_program_seeded
from agm.agl.scope import ModuleResolution
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import ConstructorRef, ScopeNode
from agm.agl.typecheck import CheckedModule
from agm.agl.typecheck.env import TypeEnvironment
from agm.agl.typecheck.program import check_program

_REPO_STDLIB_ROOT = Path(__file__).resolve().parents[2] / "stdlib"

# The cached std/core module's node ids are seeded from a high base so they
# never collide with an entry's own ids, which every call seeds from 0 (some
# tests reason about entry node ids starting at 0). New per-call node ids
# (the synthetic stdlib import, or any further module an entry explicitly
# imports) are seeded from the same high base, kept above the cached range.
_STD_CORE_SEED_START = 1_000_000

_std_core_cache: dict[ModuleId, LoadedModule] | None = None
_std_core_next_start_id: int = 0


def _roots() -> RootSet:
    return RootSet(roots=frozenset({_REPO_STDLIB_ROOT}))


def _cached_std_core() -> tuple[dict[ModuleId, LoadedModule], int]:
    """Return the process-cached ``std/core`` module plus the next free node id.

    Loaded once per process (per ``pytest -n auto`` worker) and reused for
    every call with ``default_stdlib=True``, avoiding a ~17ms reparse of
    ``std/core.agl`` on every single test.
    """
    global _std_core_cache, _std_core_next_start_id
    if _std_core_cache is None:
        priming_program, next_id = parse_program_seeded("()", start_id=_STD_CORE_SEED_START)
        _graph, next_id, new_modules = build_repl_graph(
            priming_program,
            next_id,
            path=None,
            cached={},
            roots=_roots(),
        )
        _std_core_cache = {STD_CORE_ID: new_modules[STD_CORE_ID]}
        _std_core_next_start_id = next_id
    return _std_core_cache, _std_core_next_start_id


def _build_graph(
    source: str,
    *,
    origin_path: Path | None,
    default_stdlib: bool,
) -> tuple[ModuleGraph, int | None]:
    """Build the real module graph for *source*.

    Returns the graph plus the node id ``build_repl_graph`` assigned to the
    synthetic ``std/core`` import it injected into the entry, or ``None``
    when *default_stdlib* is ``False`` (nothing was injected).
    """
    entry_program, entry_next_id = parse_program_seeded(source, start_id=0)
    if default_stdlib:
        cached, next_start_id = _cached_std_core()
    else:
        cached, next_start_id = {}, entry_next_id
    import_node_id = next_start_id if default_stdlib else None
    graph, _next_start_id, _new_modules = build_repl_graph(
        entry_program,
        next_start_id,
        path=origin_path,
        cached=cached,
        roots=_roots(),
        default_stdlib=default_stdlib,
    )
    return graph, import_node_id


def _without_synthetic_import(
    resolved: ModuleResolution, import_node_id: int | None
) -> ModuleResolution:
    """Return *resolved* with the injected ``std/core`` import dropped from view.

    Matches strictly by *import_node_id* (the id the helper itself assigned
    the synthetic import), never by position or node type, so a test's own
    ``import`` declarations are left exactly where it wrote them.
    """
    if import_node_id is None:
        return resolved
    program = resolved.program
    stripped_items = tuple(item for item in program.body.items if item.node_id != import_node_id)
    stripped_body = dataclasses.replace(program.body, items=stripped_items)
    stripped_program = dataclasses.replace(program, body=stripped_body)
    return dataclasses.replace(resolved, program=stripped_program)


def resolve_entry(
    source: str,
    *,
    parent_scope: ScopeNode | None = None,
    ambient_agents: frozenset[str] = frozenset(),
    ambient_constructor_candidates: dict[str, tuple[ConstructorRef, ...]] | None = None,
    ambient_type_names: frozenset[str] = frozenset(),
    origin_path: Path | None = None,
    default_stdlib: bool = True,
) -> ModuleResolution:
    """Resolve *source* as the entry of a real module graph.

    Builds a real :class:`~agm.agl.modules.loader.ModuleGraph` (entry plus
    ``std/core`` unless *default_stdlib* is ``False``) and runs
    :func:`~agm.agl.scope.program.resolve_program` over it, returning the
    entry module's :class:`~agm.agl.scope.symbols.ModuleResolution` with the
    injected ``std/core`` import removed from ``program.body.items`` (see
    the module docstring) — so it always mirrors the source the test wrote.

    Parameters mirror the entry-scoped parameters of ``resolve_program``:
    *parent_scope*, *ambient_agents*, *ambient_constructor_candidates*, and
    *ambient_type_names* forward to that function's ``entry_parent_scope``,
    ``ambient_agents``, ``entry_ambient_constructor_candidates``, and
    ``entry_ambient_type_names`` respectively; *origin_path* forwards to
    ``build_repl_graph``'s ``path``.

    *default_stdlib* controls whether ``std/core`` is imported into the
    entry, matching real program execution — this is ``True`` by default
    because that is what production always does. Pass ``False`` only when
    being stdlib-free is itself the point: a test that declares its own
    ``builtin`` (a bare builtin name may be declared only once per program,
    and ``std/core`` already declares every built-in name), or one that
    asserts a name is genuinely undefined with nothing else in scope. Never
    for positional convenience over ``program.body.items`` — the synthetic
    import is already invisible there regardless of this flag.
    """
    graph, import_node_id = _build_graph(
        source, origin_path=origin_path, default_stdlib=default_stdlib
    )
    resolved_program = resolve_program(
        graph,
        ambient_agents=ambient_agents,
        entry_ambient_constructor_candidates=ambient_constructor_candidates,
        entry_ambient_type_names=ambient_type_names,
        entry_parent_scope=parent_scope,
    )
    resolved = resolved_program.modules[graph.entry_id].resolved
    return _without_synthetic_import(resolved, import_node_id)


def resolve_and_check_entry(
    source: str,
    capabilities: HostCapabilities,
    *,
    parent_scope: ScopeNode | None = None,
    ambient_agents: frozenset[str] = frozenset(),
    ambient_constructor_candidates: dict[str, tuple[ConstructorRef, ...]] | None = None,
    ambient_type_names: frozenset[str] = frozenset(),
    origin_path: Path | None = None,
    seed_env: TypeEnvironment | None = None,
    default_stdlib: bool = True,
) -> CheckedModule:
    """Resolve and type-check *source* as the entry of a real module graph.

    Like :func:`resolve_entry`, but also runs
    :func:`~agm.agl.typecheck.program.check_program` over the resolved graph
    and returns the entry module's
    :class:`~agm.agl.typecheck.env.CheckedModule`. Resolution and checking
    both run with the injected ``std/core`` import present — that is the
    real graph production builds — and only the returned
    ``checked.resolved.program.body.items`` has that one synthetic item
    removed, exactly as :func:`resolve_entry` does. *capabilities* and
    *seed_env* forward to ``check_program``'s ``capabilities`` and
    ``entry_seed_env``; the remaining parameters are as in
    :func:`resolve_entry`, including the *default_stdlib* default and when
    to flip it off.
    """
    graph, import_node_id = _build_graph(
        source, origin_path=origin_path, default_stdlib=default_stdlib
    )
    resolved_program = resolve_program(
        graph,
        ambient_agents=ambient_agents,
        entry_ambient_constructor_candidates=ambient_constructor_candidates,
        entry_ambient_type_names=ambient_type_names,
        entry_parent_scope=parent_scope,
    )
    checked_program = check_program(resolved_program, capabilities, entry_seed_env=seed_env)
    checked = checked_program.modules[graph.entry_id]
    stripped_resolved = _without_synthetic_import(checked.resolved, import_node_id)
    return dataclasses.replace(checked, resolved=stripped_resolved)
