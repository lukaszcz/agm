"""Module-id to file-path resolution for the AgL module system.

This module provides two public functions:

- :func:`resolve_module` — resolve a single :class:`~agm.agl.modules.ids.ModuleId`
  to its unique canonical :class:`~pathlib.Path`.
- :func:`expand_wildcard` — expand a wildcard prefix (``foo/*``) into every
  matching :class:`~agm.agl.modules.ids.ModuleId` and its canonical path.

Both enforce **global-uniqueness**: an id that resolves to ≥2 distinct
canonical files is an :class:`~agm.agl.modules.errors.AmbiguousModule` error;
there is no first-root-wins shadowing.  Results are deterministic regardless of
the iteration order of the unordered :class:`~agm.agl.modules.roots.RootSet`.
"""

from __future__ import annotations

import os
from pathlib import Path

from agm.agl.modules.errors import AmbiguousModule, ModuleNotFound, ModulePrefixNotFound
from agm.agl.modules.ids import ModuleId
from agm.agl.modules.roots import RootSet
from agm.agl.syntax.spans import SourceSpan
from agm.core import fs


def resolve_module(
    module_id: ModuleId,
    roots: RootSet,
    *,
    span: SourceSpan | None = None,
) -> Path:
    """Resolve *module_id* to its unique canonical file path.

    Searches every loose root in *roots* for ``<root>/<module_id.relpath()>``
    and every module tree mounted for the id's leading segment for the path
    the remaining segments spell, canonicalizes each hit, and deduplicates by
    canonical identity (so the same file reached via symlinked roots or through
    both a loose root and a mount counts once).

    Parameters
    ----------
    module_id:
        The module id to resolve.
    roots:
        The set of search roots.
    span:
        Optional source span of the triggering import declaration, attached to
        any error raised.

    Returns
    -------
    Path
        The unique canonical absolute path of the module file.

    Raises
    ------
    ModuleNotFound
        When no searched directory contains a file for *module_id*.
    AmbiguousModule
        When the id resolves to ≥2 distinct canonical files.
    """
    # Map canonical path → (one of the) directory that produced it.
    # Using a dict keyed on canonical Path deduplicates symlinked copies.
    canonical_hits: dict[Path, Path] = {}

    search_roots = roots.sorted_roots()
    searched: list[Path] = list(search_roots)
    rel = module_id.relpath().replace("/", os.sep)
    for root in search_roots:
        candidate = root / rel
        if fs.exists(candidate):
            canonical_hits[candidate.resolve()] = root

    if len(module_id.segments) > 1:
        mounted_rel = _relative_path(module_id.segments[1:]) + ".agl"
        for tree in roots.module_tree_roots(module_id.segments[0]):
            searched.append(tree)
            candidate = tree / mounted_rel
            if fs.exists(candidate):
                canonical_hits[candidate.resolve()] = tree

    if not canonical_hits:
        raise ModuleNotFound(
            module_id,
            tuple(searched),
            span=span,
        )

    if len(canonical_hits) > 1:
        candidates = tuple(sorted(canonical_hits.keys()))
        raise AmbiguousModule(module_id, candidates, span=span)

    (canon,) = canonical_hits
    return canon


def expand_wildcard(
    prefix: tuple[str, ...],
    roots: RootSet,
    *,
    span: SourceSpan | None = None,
) -> dict[ModuleId, Path]:
    """Expand a wildcard prefix to all matching module ids and their canonical paths.

    Across every loose root, globs ``<root>/<prefix>.agl`` (the prefix module
    itself, if it exists) and ``<root>/<prefix>/**/*.agl`` (the full subtree).
    Across every module tree mounted for the prefix's leading segment, globs
    the same two patterns for the remaining segments — or, when the prefix is
    the mount name alone, the whole tree, since a package has no module of its
    own name.

    Each discovered file is mapped to its slash-path :class:`~agm.agl.modules.ids.ModuleId`
    via the inverse of ``ModuleId.relpath()``.  Global uniqueness is enforced: if
    the same id is found twice as distinct canonical files, an
    :class:`~agm.agl.modules.errors.AmbiguousModule` error is raised.  The
    same canonical file reached via different roots (symlinks/duplicates) is
    counted once.

    Parameters
    ----------
    prefix:
        The wildcard prefix as a tuple of segments, e.g. ``("foo", "bar")``
        for ``import foo/bar/*``.
    roots:
        The set of search roots.
    span:
        Optional source span of the triggering import declaration, attached to
        any error raised.

    Returns
    -------
    dict[ModuleId, Path]
        Mapping of every matched :class:`~agm.agl.modules.ids.ModuleId` to its
        canonical file path, **ordered by ModuleId** (lexicographic on
        ``segments``) for deterministic output.

    Raises
    ------
    ModulePrefixNotFound
        When no module matches the wildcard prefix.
    AmbiguousModule
        When any matched module id resolves to ≥2 distinct canonical files.
    """
    # Accumulated results: module_id → set of distinct canonical paths found.
    # Using a set of canonicals handles dedup of symlinked/duplicate roots.
    hits: dict[ModuleId, set[Path]] = {}

    def _record_file(file_path: Path, base: Path, lead: tuple[str, ...]) -> None:
        """Record a confirmed .agl file path, computing its ModuleId from *base*.

        *file_path* must be a file that resides under *base* and whose name
        ends with ``.agl``; these invariants are guaranteed by the caller.
        *lead* is the mount's own segment, empty for a loose root.
        """
        canon = file_path.resolve()
        parts = file_path.relative_to(base).parts
        # parts is non-empty (file_path is inside base) and its last element
        # ends with ".agl" (guaranteed by the glob pattern used to find it).
        mid = ModuleId(segments=(*lead, *parts[:-1], parts[-1][:-4]))
        hits.setdefault(mid, set()).add(canon)

    def _collect(base: Path, relative: tuple[str, ...], lead: tuple[str, ...]) -> None:
        """Glob the prefix module and its subtree beneath *base*."""
        if relative:
            # Pattern 1: <base>/<relative>.agl — the prefix module itself
            direct = base / (_relative_path(relative) + ".agl")
            if fs.is_file(direct):
                _record_file(direct, base, lead)
        # Pattern 2: <base>/<relative>/**/*.agl — the full subtree
        subtree_root = base.joinpath(*relative)
        if fs.is_dir(subtree_root):
            for file_path in fs.rglob(subtree_root, "*.agl"):
                if fs.is_file(file_path):
                    _record_file(file_path, base, lead)

    for root in roots.sorted_roots():
        _collect(root, prefix, ())

    for tree in roots.module_tree_roots(prefix[0]):
        _collect(tree, prefix[1:], prefix[:1])

    if not hits:
        raise ModulePrefixNotFound(prefix, span=span)

    # Enforce global uniqueness: any id with ≥2 distinct canonical files is ambiguous.
    for mid, canons in hits.items():
        if len(canons) > 1:
            candidates = tuple(sorted(canons))
            raise AmbiguousModule(mid, candidates, span=span)

    # Build the result dict ordered by ModuleId (lexicographic on segments).
    pairs: list[tuple[ModuleId, Path]] = [(mid, next(iter(canons))) for mid, canons in hits.items()]

    def _sort_key(pair: tuple[ModuleId, Path]) -> tuple[str, ...]:
        return pair[0].segments

    pairs.sort(key=_sort_key)
    return dict(pairs)


def _relative_path(segments: tuple[str, ...]) -> str:
    """Join module-id segments into an os-specific relative path without a suffix."""
    return os.sep.join(segments)
