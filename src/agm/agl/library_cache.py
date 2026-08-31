"""Process-global reuse of the artifacts a compilation derives from the library.

Every compilation loads the whole standard library before it can check a line of
the program in front of it, and re-deriving those modules' artifacts dominates
the cost of a short program: for a few-line entry they are almost all of it.
The passes already know how to skip that work -- :func:`~agm.agl.scope.program.
resolve_program` takes ``cached_modules``, :func:`~agm.agl.typecheck.program.
check_program` takes ``cached_checked_modules``, and
:func:`~agm.agl.matchcompile.compile_program_matches` takes ``cached_sites`` --
but only a REPL session, which holds its own image across entries, ever
supplied them.  This module is the same image for every other caller: one
compilation's library artifacts, kept for the next one in the same process.

What makes an artifact reusable is that it is a pure function of the library
modules it was derived from, never of the entry.  So an artifact is retained
alongside the *loaded modules the derivation could read* -- the module itself,
everything reachable from it through the graph's import and export edges, and
the ambient builtin-method modules every module can see -- and it is served
again only while every one of those is the very same object.  The parse and
infix-resolution caches in :mod:`agm.agl.modules.parsed_module_cache` are what
make that possible: they are why loading the same library file twice yields the
same object rather than an equal one.

That identity condition is the whole condition, which is why nothing here is
keyed by root set.  A root set that resolves a library import to a different
file yields a different loaded module and misses; one that merely adds
unrelated user roots -- a temporary project directory, a package mount -- leaves
every library module untouched and hits.  Host capabilities are not derivable
from modules, so they key the artifacts that are checked against them.

Only standard-library modules are retained.  User and package modules change
far more freely during a process's life, and the standard library is where the
whole win is.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agm.agl.capabilities import HostCapabilities
    from agm.agl.matchcompile import CachedModuleSites
    from agm.agl.modules.ids import ModuleId
    from agm.agl.modules.loader import LoadedModule, ModuleGraph
    from agm.agl.scope.program import ResolvedModule
    from agm.agl.typecheck.env import CheckedModule

# One entry per library module per distinct library a process compiles against.
# A standard library is a few dozen modules, so this holds several at once while
# staying a fixed, small amount of retained memory.
_CAPACITY = 1024

# Everything a library module's artifacts could have been derived from.
Sources = tuple["LoadedModule", ...]

# (module id, capability signature) for artifacts checked against capabilities,
# and (module id,) for those derived from the modules alone.
_Key = tuple[object, ...]


def capability_signature(capabilities: HostCapabilities) -> tuple[object, ...]:
    """Return a hashable stand-in for the capabilities a module was checked under."""
    return (
        capabilities.supports_shell_exec,
        capabilities.supports_extern,
        tuple(
            sorted((name, tuple(sorted(kinds))) for name, kinds in capabilities.codec_kinds.items())
        ),
    )


@dataclass(frozen=True, slots=True)
class _Entry[V]:
    """One retained artifact and the loaded modules its derivation could read."""

    sources: Sources
    artifact: V


class _ArtifactStore[V]:
    """A bounded, least-recently-used store of one pass's library artifacts."""

    def __init__(self, *, capacity: int = _CAPACITY) -> None:
        self._capacity = capacity
        self._entries: OrderedDict[_Key, _Entry[V]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: _Key, sources: Sources) -> V | None:
        """Return the artifact retained under *key*, if *sources* are unchanged."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or not _same_sources(entry.sources, sources):
                return None
            self._entries.move_to_end(key)
            return entry.artifact

    def put(self, key: _Key, sources: Sources, artifact: V) -> None:
        """Retain *artifact* under *key* for as long as *sources* stay identical."""
        with self._lock:
            self._entries[key] = _Entry(sources, artifact)
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Drop every retained artifact."""
        with self._lock:
            self._entries.clear()


def _same_sources(retained: Sources, current: Sources) -> bool:
    """Return whether two source tuples name the very same loaded modules."""
    return len(retained) == len(current) and all(
        left is right for left, right in zip(retained, current, strict=True)
    )


# What one compilation's library modules were derived from, computed once by the
# pass that both consults and refreshes the image.
LibrarySources = Mapping["ModuleId", Sources]

_RESOLVED: _ArtifactStore["ResolvedModule"] = _ArtifactStore()
_CHECKED: _ArtifactStore["CheckedModule"] = _ArtifactStore()
_SITES: _ArtifactStore["CachedModuleSites"] = _ArtifactStore()


def library_module_sources(graph: ModuleGraph) -> dict[ModuleId, Sources]:
    """Return, per standard-library module, everything its artifacts could read.

    A module's own artifacts depend on it, on every module reachable from it
    through the graph's dependency edges (whose exports decide what its imports
    name), and on the ambient builtin-method modules, which contribute methods
    to every module without appearing as a dependency edge.

    A pass derives this once and uses it twice -- to look the image up, and to
    refresh it with what the pass produced.
    """
    sources: dict[ModuleId, Sources] = {}
    for module_id, loaded in graph.modules.items():
        if loaded.path is None or not graph.roots.is_standard_library_path(loaded.path):
            continue
        reachable = _reachable(graph, module_id) | set(graph.ambient_modules)
        sources[module_id] = tuple(
            graph.modules[reached]
            for reached in sorted(reachable, key=_ordering_key)
            if reached in graph.modules
        )
    return sources


def _ordering_key(module_id: ModuleId) -> str:
    """Return a stable ordering key, so two graphs list the same modules alike."""
    return module_id.path_str()


def _reachable(graph: ModuleGraph, start: ModuleId) -> set[ModuleId]:
    """Return *start* and every module reachable from it through dependency edges."""
    seen = {start}
    frontier = [start]
    while frontier:
        for neighbour in graph.adjacency.get(frontier.pop(), ()):
            if neighbour not in seen:
                seen.add(neighbour)
                frontier.append(neighbour)
    return seen


def resolved_library_modules(library: LibrarySources) -> dict[ModuleId, ResolvedModule]:
    """Return the resolved modules retained for this library."""
    return _served(_RESOLVED, library, ())


def retain_resolved_library_modules(
    library: LibrarySources, modules: Mapping[ModuleId, ResolvedModule]
) -> None:
    """Retain a compilation's resolved library modules for the next one."""
    _retain(_RESOLVED, library, (), modules)


def checked_library_modules(
    library: LibrarySources, capabilities: HostCapabilities
) -> dict[ModuleId, CheckedModule]:
    """Return the checked modules retained for this library."""
    return _served(_CHECKED, library, (capability_signature(capabilities),))


def retain_checked_library_modules(
    library: LibrarySources,
    capabilities: HostCapabilities,
    modules: Mapping[ModuleId, CheckedModule],
) -> None:
    """Retain a compilation's checked library modules for the next one."""
    _retain(_CHECKED, library, (capability_signature(capabilities),), modules)


def library_match_sites(
    library: LibrarySources, capabilities: HostCapabilities
) -> dict[ModuleId, CachedModuleSites]:
    """Return the compiled match sites retained for this library."""
    return _served(_SITES, library, (capability_signature(capabilities),))


def retain_library_match_sites(
    library: LibrarySources,
    capabilities: HostCapabilities,
    sites: Mapping[ModuleId, CachedModuleSites],
) -> None:
    """Retain a compilation's compiled library match sites for the next one."""
    _retain(_SITES, library, (capability_signature(capabilities),), sites)


def _served[V](
    store: _ArtifactStore[V], library: LibrarySources, discriminator: tuple[object, ...]
) -> dict[ModuleId, V]:
    """Collect every artifact this library's modules still qualify for."""
    served: dict[ModuleId, V] = {}
    for module_id, sources in library.items():
        artifact = store.get((module_id, *discriminator), sources)
        if artifact is not None:
            served[module_id] = artifact
    return served


def _retain[V](
    store: _ArtifactStore[V],
    library: LibrarySources,
    discriminator: tuple[object, ...],
    artifacts: Mapping[ModuleId, V],
) -> None:
    """Retain every artifact belonging to one of this library's modules."""
    for module_id, artifact in artifacts.items():
        sources = library.get(module_id)
        if sources is not None:
            store.put((module_id, *discriminator), sources, artifact)


def clear_library_image_cache() -> None:
    """Discard every retained library artifact.

    Hosts that replace a standard library in place, and tests that must observe
    a cold compilation, call this; ordinary compilation never needs it, since a
    replaced file is reparsed and its artifacts then fail their own identity
    condition.
    """
    _RESOLVED.clear()
    _CHECKED.clear()
    _SITES.clear()
