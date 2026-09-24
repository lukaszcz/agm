"""Process-global cache of parsed modules.

Every AgL compilation — ``agm exec``, ``agm check``, a REPL start, package
discipline validation — re-lexes and re-parses every module it loads, which
dominates the cost of a short program. This cache keeps each file's parsed
module for the life of the process so later compilations skip the parse.

Nothing distinguishes the standard library here. A module is cacheable because
its source is unchanged, not because of the root it was found under, so
ordinary user and package modules are served on exactly the same terms.

Imported modules draw node ids from content-addressed namespaces above
:data:`RESERVED_NODE_ID_BASE`. A module's path, logical identity, source and
prelude mode select its namespace; allocation within it is local. The same
module version therefore has the same declaration ids across processes and
import orders, while edits and distinct modules remain disjoint from it and
from ordinary entry allocation.

An entry is keyed by canonical path, module id, and whether the loader injects
the standard-library prelude, and it is validated against the source text
itself rather than against the file's stat metadata. Metadata cannot separate
an in-place rewrite of the same length under a preserved modification time,
and a process that both writes and compiles modules — a REPL session editing a
file, a test writing into a temporary tree — reaches exactly that case. The
parsed module already carries the text it was parsed from, so comparing it
costs a read and no extra storage, and the read is the one the parse needed
anyway: the cache hands its text to the builder on a miss.

Artifacts derived from a parsed module are cached beside it, through
:class:`ModuleDerivationCache`, and served only to the very module they were
derived from. Infix-chain resolution is one: it rewrites a module's
program against the operators visible to it, so a module re-resolved
per compilation would hand every later pass a structurally equal but *fresh*
program object, defeating the identity-keyed reuse guards in scope resolution
and type checking. :func:`cached_infix_resolution` therefore memoizes the
rewrite on the parsed module together with the exact operator tables it was
resolved against, so a later graph that presents the same module the same
operators reuses the very program object it produced before, and one that
presents different operators resolves afresh. Locating the named scope that
owns each raw chain, which the rewrite needs before it can consult that memo,
is cached the same way.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from agm.agl.modules import disk_cache
from agm.core import fs
from agm.util.text import normalize_newlines

if TYPE_CHECKING:
    from agm.agl.modules.ids import ModuleId
    from agm.agl.modules.loader import LoadedModule

# The first node id reserved for cached modules. Ordinary allocation starts at
# zero and grows by one per AST node, so no compilation reaches this band;
# imported declarations use content-addressed integer namespaces above it.
RESERVED_NODE_ID_BASE = 1 << 40

# An entry holds one module's syntax tree, so the bound trades memory for
# reparses. A long-lived process sees the standard library plus whatever
# projects it compiles; this holds several of both at once without letting a
# session that walks over many trees grow without limit.
_DEFAULT_CAPACITY = 512

# (canonical path, module id, prelude injected)
_CacheKey = tuple[str, "ModuleId", bool]

# Given the first node id it may use and the module's source text, returns the
# parsed module and the first node id it did not use.
ModuleBuilder = Callable[[int, str], "tuple[LoadedModule, int]"]

# Everything infix-chain resolution reads besides the program itself: the
# operator tables and conflicting-operator sets the loader derived for one
# module from the graph around it.
InfixSignature = tuple[object, ...]

# Rewrites one module's infix chains, returning its resolved form.
InfixResolver = Callable[[], "LoadedModule"]


@dataclass(frozen=True, slots=True)
class _DerivedEntry[V]:
    """One artifact derived from a parsed module, and the module it came from.

    *source* pins the exact parsed module the derivation consumed, so an entry
    is served only to that same object.  A reparse (the file changed, or the
    cache was cleared) produces a different module and misses.
    """

    source: LoadedModule
    value: V


def _companion_intact(module: LoadedModule) -> bool:
    """Return whether a cached module's Python companion still exists.

    Parsing a module verifies the companion an ``extern def`` requires, so a
    hit re-verifies it; otherwise a companion removed after the first load
    would go unreported for the rest of the process.
    """
    return module.companion_path is None or fs.is_file(module.companion_path)


def _emits_lexical_advisories(source_text: str) -> bool:
    """Return whether parsing *source_text* would emit TAB advisories.

    Unlike spaced-qualifier advisories, which the loader records on the parsed
    module, TAB advisories are deposited into whichever collector is active
    during the parse. A module that produces them is therefore never cached,
    so every compilation that loads it still surfaces them.
    """
    return "\t" in source_text


def module_node_id_base(
    module_id: ModuleId, path: Path, source_text: str, default_stdlib: bool
) -> int:
    """Allocate a reproducible node namespace for a module's source version."""
    identity = repr(((str(path), module_id, default_stdlib), source_text)).encode()
    return (int.from_bytes(hashlib.sha256(identity).digest()[:16]) + 1) * RESERVED_NODE_ID_BASE


class ParsedModuleCache:
    """A bounded store of parsed modules keyed by path, id, and prelude.

    Entries are evicted least-recently-used first. Source-version namespaces
    remain stable across eviction and reopening, independently of which other
    graphs a host retains.
    """

    def __init__(self, *, capacity: int = _DEFAULT_CAPACITY) -> None:
        self._capacity = capacity
        self._entries: OrderedDict[_CacheKey, LoadedModule] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_build(
        self,
        module_id: ModuleId,
        path: Path,
        *,
        default_stdlib: bool,
        build: ModuleBuilder,
    ) -> LoadedModule:
        """Return *path*'s parsed module, calling *build* only on a miss.

        The file is read here and its text compared against the text the cached
        module was parsed from, so an entry is served only for source that is
        byte-for-byte what produced it. On a miss that same text is handed to
        *build*, which is seeded from the reserved band, so the caller's own
        node-id counter is neither read nor advanced.
        """
        key = (str(path), module_id, default_stdlib)
        source_text = normalize_newlines(fs.read_text(path))
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.source_text == source_text and _companion_intact(entry):
                self._entries.move_to_end(key)
                return entry
            cacheable = not _emits_lexical_advisories(source_text)
            persistent = cacheable
            # A content-addressed namespace survives process and import-order
            # changes. Leave a 32-bit local allocation range for each version.
            start_id = module_node_id_base(module_id, path, source_text, default_stdlib)
            restored = (
                disk_cache.load(module_id, path, source_text, start_id, default_stdlib)
                if persistent
                else None
            )
            if restored is None:
                module, next_id = build(start_id, source_text)
                if persistent:
                    disk_cache.save(
                        module, next_id, start_id=start_id, default_stdlib=default_stdlib
                    )
            else:
                module, _next_id = restored
            if cacheable:
                self._entries[key] = module
                self._entries.move_to_end(key)
                while len(self._entries) > self._capacity:
                    self._entries.popitem(last=False)
            return module

    def clear(self) -> None:
        """Drop every cached module."""
        with self._lock:
            self._entries.clear()


class ModuleDerivationCache[K, V]:
    """A bounded store of artifacts derived from one parsed library module.

    Keyed by the module's file identity and whatever else the derivation reads
    besides the module itself, and served only to the very parsed module the
    derivation consumed, so a reparse or a differing input derives afresh.
    """

    def __init__(self, *, capacity: int = _DEFAULT_CAPACITY) -> None:
        self._capacity = capacity
        self._entries: OrderedDict[tuple[str, ModuleId, K], _DerivedEntry[V]] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_build(
        self,
        module: LoadedModule,
        *,
        key: K,
        build: Callable[[], V],
    ) -> V:
        """Return the artifact derived from *module*, calling *build* on a miss."""
        entry_key = (str(module.path), module.module_id, key)
        with self._lock:
            entry = self._entries.get(entry_key)
            if entry is not None and entry.source is module:
                self._entries.move_to_end(entry_key)
                return entry.value
            value = build()
            self._entries[entry_key] = _DerivedEntry(module, value)
            self._entries.move_to_end(entry_key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
            return value

    def clear(self) -> None:
        """Drop every memoized artifact."""
        with self._lock:
            self._entries.clear()


_PARSED_MODULES = ParsedModuleCache()
_INFIX_RESOLUTIONS: ModuleDerivationCache[InfixSignature, LoadedModule] = ModuleDerivationCache()
_CHAIN_SCOPE_PATHS: ModuleDerivationCache[None, "dict[int, tuple[str, ...]]"] = (
    ModuleDerivationCache()
)


def cached_infix_resolution(
    module: LoadedModule,
    *,
    signature: InfixSignature,
    resolve: InfixResolver,
) -> LoadedModule:
    """Serve one module's infix-resolved form.

    The loader rewrites every module's infix chains against the operators
    visible to it.  For an unchanged module that rewrite is the same on every
    compilation that presents the same operators, and repeating it would return
    a fresh program object each time — equal to the last one, but not the same,
    which is what the scope and type-check reuse guards key on.
    """
    return _INFIX_RESOLUTIONS.get_or_build(module, key=signature, build=resolve)


def cached_chain_scope_paths(
    module: LoadedModule,
    *,
    build: Callable[[], "dict[int, tuple[str, ...]]"],
) -> "dict[int, tuple[str, ...]]":
    """Serve the named scope owning each raw infix chain of a module.

    Locating them walks the module's whole syntax tree, and the answer depends
    on nothing but that tree, so it is derived once per parse rather than once
    per compilation.
    """
    return _CHAIN_SCOPE_PATHS.get_or_build(module, key=None, build=build)


def cached_parsed_module(
    module_id: ModuleId,
    path: Path,
    *,
    default_stdlib: bool,
    build: ModuleBuilder,
) -> LoadedModule:
    """Serve one module's parse from the process-global cache."""
    return _PARSED_MODULES.get_or_build(module_id, path, default_stdlib=default_stdlib, build=build)


def clear_parsed_module_cache() -> None:
    """Discard every parsed module the process has cached.

    Hosts that replace modules in place, and tests that must observe a cold
    parse, call this; ordinary compilation never needs it, since changed source
    simply misses.
    """
    _PARSED_MODULES.clear()
    _INFIX_RESOLUTIONS.clear()
    _CHAIN_SCOPE_PATHS.clear()
