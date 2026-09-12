"""Reusable module artifacts in memory and across CLI processes.

Resolution, checking, and match compilation retain each module alongside its
transitive import/export dependencies. Memory hits
require identical loaded sources; disk hits validate source-derived identities
and the compiler version. Modules sharing a dependency closure share a persisted
stage artifact. Restored stages anchor their provenance to the current compilation.

Checked modules publish closed type and function interfaces for importers.
Capabilities distinguish checked artifacts and IR; the lowerer adds its resource
and contract context. Entry modules are never retained; non-entry cycle members may be retained
against the exact entry source. Runtime state and companion callables are never cached here.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

from agm.agl import artifact_serialization
from agm.agl.modules.ids import ModuleId

if TYPE_CHECKING:
    from agm.agl.capabilities import HostCapabilities
    from agm.agl.lower.module import LoweredModule
    from agm.agl.matchcompile import CachedModuleSites
    from agm.agl.modules.loader import LoadedModule, ModuleGraph
    from agm.agl.scope.program import ResolvedModule
    from agm.agl.typecheck.env import CheckedModule, CheckedModuleImage

# One entry per module per distinct set of sources a process compiles against.
# A standard library is a few dozen modules and a project rarely many more, so
# this holds several of both at once while staying a fixed, small amount of
# retained memory.
_CAPACITY = 1024

# Everything a module's artifacts could have been derived from.
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
    """A bounded, least-recently-used store of one pass's artifacts.

    A store's memory form and its disk form need not coincide: the ``expected``
    type :func:`_served` validates a restored payload against, and the
    ``disk_form`` :func:`_retain` writes, are the two halves of one contract
    each owning function states together (see :func:`retained_checked_modules`
    and :func:`retain_checked_modules`, whose disk form is a data-only image).
    """

    def __init__(self, *, capacity: int = _CAPACITY, kind: str = "") -> None:
        self.kind = kind
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


# What one compilation's retainable modules were derived from, computed once by
# the pass that both consults and refreshes the retained artifacts.
RetainedSources = Mapping["ModuleId", Sources]


_RESOLVED: _ArtifactStore["ResolvedModule"] = _ArtifactStore(kind="scope")
# Holds full CheckedModule objects for in-process reuse; a disk-served entry
# is a data-only CheckedModuleImage until check_program rehydrates and retains
# the full object (see retained_checked_modules / retain_checked_modules).
_CHECKED: _ArtifactStore["CheckedModule | CheckedModuleImage"] = _ArtifactStore(kind="checked")
_SITES: _ArtifactStore["CachedModuleSites"] = _ArtifactStore(kind="matches")


_LOWERED: _ArtifactStore["LoweredModule"] = _ArtifactStore()


def retained_lowered_module(key: bytes) -> LoweredModule | None:
    """Return linked module data retained under its full compilation identity."""
    return _LOWERED.get((key,), ())


def retain_lowered_module(key: bytes, module: LoweredModule) -> None:
    """Keep immutable module IR for subsequent compilations in this process."""
    _LOWERED.put((key,), (), module)


def retained_module_sources(graph: ModuleGraph) -> dict[ModuleId, Sources]:
    """Return, per retainable module, everything its artifacts could read.

    A module's own artifacts depend on it and every module reachable from it
    through the graph's dependency edges, whose exports decide what its imports
    name.

    A pass derives this once and uses it twice -- to look up what is
    retained, and to refresh it with what the pass produced.
    """
    sources: dict[ModuleId, Sources] = {}
    for module_id, loaded in graph.modules.items():
        if loaded.path is None or module_id == graph.entry_id:
            continue
        reachable = _reachable(graph, module_id)
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


def retained_resolved_modules(retainable: RetainedSources) -> dict[ModuleId, ResolvedModule]:
    """Return the resolved modules retained for these sources."""
    from agm.agl.scope.program import ResolvedModule

    return _served(_RESOLVED, retainable, (), ResolvedModule)


def retain_resolved_modules(
    retainable: RetainedSources, modules: Mapping[ModuleId, ResolvedModule]
) -> None:
    """Retain a compilation's resolved modules for the next one."""
    _retain(_RESOLVED, retainable, (), modules)


def retained_checked_modules(
    retainable: RetainedSources, capabilities: HostCapabilities
) -> dict[ModuleId, CheckedModule | CheckedModuleImage]:
    """Return the checked modules retained for these sources.

    A module is a full ``CheckedModule`` iff some compilation in this process
    checked it, or rehydrated and retained it; it is a data-only
    ``CheckedModuleImage`` while it has only ever come off disk in this
    process (every ``entry_seed_env``/REPL compilation skips retention, so an
    image can outlive many compilations). ``check_program`` rehydrates an
    image onto the environment its own preparation phases build.
    """
    from agm.agl.typecheck.env import CheckedModule, CheckedModuleImage

    signature = capability_signature(capabilities)
    # Disk holds only images; a full object in `served` came from memory.
    served = _served(_CHECKED, retainable, (signature,), CheckedModuleImage)
    for mid, module in served.items():
        # An image has no `resolved` to refresh: it is not yet a CheckedModule,
        # and rehydration always attaches the current compilation's resolved
        # module directly (see CheckedModuleImage.rehydrate).
        if not isinstance(module, CheckedModule):
            continue
        resolved = _RESOLVED.get((mid,), retainable[mid])
        if resolved is not None and module.resolved is not resolved.resolved:
            served[mid] = replace(module, resolved=resolved.resolved)
            _CHECKED.put((mid, signature), retainable[mid], served[mid])
    return served


def retain_checked_modules(
    retainable: RetainedSources,
    capabilities: HostCapabilities,
    modules: Mapping[ModuleId, CheckedModule],
) -> None:
    """Retain a compilation's checked modules for the next one.

    Memory keeps the full objects; disk keeps only their data-only images, so
    a restored artifact costs a replay of the module's own facts rather than a
    copy of the whole-program type environment.
    """
    # `.image()` calls `own_facts()`, which raises if no own-facts journal was
    # ever started. Unreachable here: every member of `modules` is freshly
    # checked (Phase 4 opens a journal before its body check), rehydrated
    # (`rehydrate` replays into a fresh journal), or an in-process journaled
    # object carried over from an earlier compilation.
    _retain(
        _CHECKED,
        retainable,
        (capability_signature(capabilities),),
        modules,
        disk_form=lambda mid: modules[mid].image(),
    )


def retained_match_sites(
    retainable: RetainedSources, capabilities: HostCapabilities
) -> dict[ModuleId, CachedModuleSites]:
    """Return the compiled match sites retained for these sources."""
    from agm.agl.matchcompile import CachedModuleSites
    from agm.agl.typecheck.env import CheckedModule

    signature = capability_signature(capabilities)
    served = _served(_SITES, retainable, (signature,), CachedModuleSites)
    for mid, sites in served.items():
        checked = _CHECKED.get((mid, signature), retainable[mid])
        # A stale or bare-image `checked` just misses `compile_program_matches`'s
        # `cached.owner is checked_module` identity gate -- never a wrong reuse.
        if isinstance(checked, CheckedModule) and sites.owner is not checked:
            served[mid] = replace(sites, owner=checked)
            _SITES.put((mid, signature), retainable[mid], served[mid])
    return served


def retain_match_sites(
    retainable: RetainedSources,
    capabilities: HostCapabilities,
    sites: Mapping[ModuleId, CachedModuleSites],
) -> None:
    """Retain a compilation's compiled match sites for the next one."""
    _retain(_SITES, retainable, (capability_signature(capabilities),), sites)


def _served[V](
    store: _ArtifactStore[V],
    retainable: RetainedSources,
    discriminator: tuple[object, ...],
    expected: type[V],
) -> dict[ModuleId, V]:
    """Collect every artifact these modules still qualify for.

    *expected* is the store's disk form, which a payload must be an instance
    of to be admitted -- the decoding half of the contract whose encoding half
    is :func:`_retain`'s ``disk_form``.

    Admission checks shape (``isinstance``) and provenance (``_disk_key``'s
    content digest, mixed with ``_compiler_digest()``'s code fingerprint at
    ``artifact_entry``), never the fields inside an admitted object: the
    cache directory is the user's own disposable store, not untrusted input.
    """
    for group in _groups(retainable):
        sources = retainable[group[0]]
        if all(store.get((mid, *discriminator), retainable[mid]) is not None for mid in group):
            continue
        payload = artifact_serialization.load(
            _disk_key(sources, discriminator),
            store.kind,
            _anchors(store.kind, sources, discriminator, retainable),
        )
        if isinstance(payload, dict):
            artifacts = cast(dict[object, object], payload)
            for mid in group:
                artifact = artifacts.get(mid)
                if isinstance(artifact, expected):
                    store.put((mid, *discriminator), retainable[mid], artifact)
    served: dict[ModuleId, V] = {}
    for module_id, sources in retainable.items():
        artifact = store.get((module_id, *discriminator), sources)
        if artifact is not None:
            served[module_id] = artifact
    return served


def _retain[V](
    store: _ArtifactStore[V],
    retainable: RetainedSources,
    discriminator: tuple[object, ...],
    artifacts: Mapping[ModuleId, V],
    *,
    disk_form: Callable[[ModuleId], object] | None = None,
) -> None:
    """Retain every artifact belonging to one of these modules.

    Memory always keeps *artifacts* themselves; *disk_form*, keyed by module
    id rather than by value, lets a caller persist a smaller, data-only
    stand-in instead (checked modules persist their ``CheckedModuleImage``,
    see :func:`retain_checked_modules`). It belongs on the call, not on the
    store: a store-level disk type erases the concrete *expected* type
    :func:`_served`'s ``isinstance`` narrowing depends on, and a callable
    keyed by module id keeps each caller's mapping visible to the coverage
    gate.
    """
    for group in _groups(retainable):
        sources = retainable[group[0]]
        members = {mid: artifacts[mid] for mid in group if mid in artifacts}
        changed = any(store.get((mid, *discriminator), sources) is None for mid in members)
        for mid, artifact in members.items():
            store.put((mid, *discriminator), sources, artifact)
        if changed:
            artifact_serialization.save(
                _disk_key(sources, discriminator),
                store.kind,
                {mid: (members[mid] if disk_form is None else disk_form(mid)) for mid in members},
                _anchors(store.kind, sources, discriminator, retainable),
            )


def _anchors(
    kind: str, sources: Sources, discriminator: tuple[object, ...], retainable: RetainedSources
) -> tuple[object, ...]:
    """Return this stage's positional anchor table, indexed by persistent id.

    The layout cannot differ between a save and a later load of the same
    entry: `retain_resolved_modules` populates `_RESOLVED` for every
    retainable module before any `checked` anchor build runs here, and
    entry-ness moves a module's node ids between the reserved band and
    ordinary allocation (`modules/parsed_module_cache.py`), so any change to
    group membership breaks `_disk_key` first. Were a shorter tuple ever to
    reach a load regardless, an out-of-range persistent id makes
    `_Reader.persistent_load` raise and the load reports a miss. An in-range
    id shifted onto the wrong anchor is the hazard this table cannot rule
    out itself. For `matches` the identity gate downstream does --
    `compile_program_matches`'s `cached.owner is checked_module` check. For
    `checked` it is the layout argument above: the anchored objects are the
    retained `ResolvedModule`s of the very group `_disk_key` keys on.
    """
    from agm.agl.typecheck.env import CheckedModule

    anchors: list[object] = [module.program for module in sources]
    for module in sources:
        retained = retainable.get(module.module_id)
        if retained is None:
            continue
        if kind == "checked":
            resolved = _RESOLVED.get((module.module_id,), retained)
            if resolved is not None:
                anchors.append(resolved.resolved)
        elif kind == "matches":
            checked = _CHECKED.get((module.module_id, *discriminator), retained)
            # A stale or bare-image `checked` just misses the same identity
            # gate `retained_match_sites` relies on -- never a wrong reuse.
            if isinstance(checked, CheckedModule):
                anchors.extend((checked, checked.type_env.type_table))
    return tuple(anchors)


def _groups(retainable: RetainedSources) -> tuple[tuple[ModuleId, ...], ...]:
    groups: dict[tuple[ModuleId, ...], list[ModuleId]] = {}
    for mid, sources in retainable.items():
        groups.setdefault(tuple(m.module_id for m in sources), []).append(mid)
    return tuple(tuple(group) for group in groups.values())


_FINGERPRINTS: _ArtifactStore[bytes] = _ArtifactStore()


def _disk_key(sources: Sources, discriminator: tuple[object, ...]) -> bytes:
    digest = hashlib.sha256(repr(discriminator).encode())
    for module in sources:
        key = (module.module_id,)
        fingerprint = _FINGERPRINTS.get(key, (module,))
        if fingerprint is None:
            fingerprint = hashlib.sha256(repr(module).encode()).digest()
            _FINGERPRINTS.put(key, (module,), fingerprint)
        digest.update(fingerprint)
    return digest.digest()


def module_fingerprints(
    retainable: RetainedSources, capabilities: HostCapabilities
) -> dict[ModuleId, bytes]:
    """Dependency identities used to validate independently lowered modules."""
    return {
        mid: _disk_key(sources, (mid, capability_signature(capabilities)))
        for mid, sources in retainable.items()
    }


def clear_retained_artifacts() -> None:
    """Discard every retained artifact.

    Hosts that replace modules in place, and tests that must observe
    a cold compilation, call this; ordinary compilation never needs it, since a
    replaced file is reparsed and its artifacts then fail their own identity
    condition.
    """
    _RESOLVED.clear()
    _CHECKED.clear()
    _SITES.clear()
    _FINGERPRINTS.clear()
    _LOWERED.clear()
