"""Process-global cache of parsed standard-library modules.

Every AgL compilation — ``agm exec``, ``agm check``, a REPL start, package
discipline validation — loads the whole standard library before it can check a
single line, and re-lexing and re-parsing those files dominates the cost of a
short program. This cache keeps each standard-library file's parsed module for
the life of the process so later compilations skip the parse entirely.

Node ids are allocated across a whole module graph from a counter that starts
at zero, so a module parsed under one compilation's seed cannot simply be
reused under another's without colliding with it. Cached modules therefore
draw their ids from a reserved band far above anything ordinary allocation
reaches (:data:`RESERVED_NODE_ID_BASE`): a compilation served from here leaves
its own counter untouched, and the two id ranges stay disjoint by
construction.

Only modules resolved under a graph's standard-library roots are cached. User
and package modules change far more freely during a process's life — a REPL
session editing a file, a host writing modules into a temporary tree — and the
standard library is where the whole win is, so the loader offers nothing else.

An entry is keyed by canonical path, module id, and whether the loader injects
the standard-library prelude, and it is served only while the file's
filesystem identity (modification time, size, and inode) is unchanged, so a
standard library replaced under a running process is never served stale. The
modules a cache holds are the loader's *pre-infix-resolution* modules: infix
chains still resolve per graph, against the operators visible in that graph.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from agm.core import fs

if TYPE_CHECKING:
    from agm.agl.modules.ids import ModuleId
    from agm.agl.modules.loader import LoadedModule

# The first node id reserved for cached modules. Ordinary allocation starts at
# zero and grows by one per AST node, so no compilation reaches this band;
# every id in it is still a small Python int.
RESERVED_NODE_ID_BASE = 1 << 40

# A long-lived process sees only a handful of standard-library roots (a test
# session, a host validating packages against different libraries), so a small
# bound keeps the cache from growing without limit while holding several
# complete libraries at once.
_DEFAULT_CAPACITY = 128

# (canonical path, module id, prelude injected)
_CacheKey = tuple[str, "ModuleId", bool]

# (modification time, size, inode)
_IdentityStamp = tuple[int, int, int]

# Given the first node id it may use, returns the parsed module and the first
# node id it did not use.
ModuleBuilder = Callable[[int], "tuple[LoadedModule, int]"]


@dataclass(frozen=True, slots=True)
class _Entry:
    """One cached module and the file identity it was parsed from."""

    stamp: _IdentityStamp
    module: LoadedModule


def _identity_stamp(path: Path) -> _IdentityStamp:
    """Return *path*'s filesystem identity stamp."""
    info = fs.stat(path)
    return (info.st_mtime_ns, info.st_size, info.st_ino)


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


class ParsedModuleCache:
    """A bounded store of parsed modules keyed by file identity.

    Entries are evicted least-recently-used first. The reserved id counter is
    never rewound, including by :meth:`clear`, because a graph assembled
    before an eviction may still hold the modules that carry those ids.
    """

    def __init__(self, *, capacity: int = _DEFAULT_CAPACITY) -> None:
        self._capacity = capacity
        self._entries: OrderedDict[_CacheKey, _Entry] = OrderedDict()
        self._next_node_id = RESERVED_NODE_ID_BASE
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

        On a miss *build* is seeded from the reserved band, so the caller's own
        node-id counter is neither read nor advanced.
        """
        key = (str(path), module_id, default_stdlib)
        with self._lock:
            stamp = _identity_stamp(path)
            entry = self._entries.get(key)
            if entry is not None and entry.stamp == stamp and _companion_intact(entry.module):
                self._entries.move_to_end(key)
                return entry.module
            module, self._next_node_id = build(self._next_node_id)
            if not _emits_lexical_advisories(module.source_text):
                self._entries[key] = _Entry(stamp, module)
                self._entries.move_to_end(key)
                while len(self._entries) > self._capacity:
                    self._entries.popitem(last=False)
            return module

    def clear(self) -> None:
        """Drop every cached module."""
        with self._lock:
            self._entries.clear()


_PARSED_MODULES = ParsedModuleCache()


def cached_library_module(
    module_id: ModuleId,
    path: Path,
    *,
    default_stdlib: bool,
    build: ModuleBuilder,
) -> LoadedModule:
    """Serve one standard-library module from the process-global cache."""
    return _PARSED_MODULES.get_or_build(module_id, path, default_stdlib=default_stdlib, build=build)


def clear_parsed_module_cache() -> None:
    """Discard every parsed module the process has cached.

    Hosts that replace a standard library in place, and tests that must
    observe a cold parse, call this; ordinary compilation never needs it,
    since a replaced file is detected by its identity stamp.
    """
    _PARSED_MODULES.clear()
