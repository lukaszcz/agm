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

    Searches every root in *roots* for ``<root>/<module_id.relpath()>``,
    canonicalizes each hit, and deduplicates by canonical identity (so the
    same file reached via symlinked roots counts once).

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
        When no root contains a file for *module_id*.
    AmbiguousModule
        When the id resolves to ≥2 distinct canonical files.
    """
    # Map canonical path → (one of the) root that produced it.
    # Using a dict keyed on canonical Path deduplicates symlinked copies.
    canonical_hits: dict[Path, Path] = {}

    rel = module_id.relpath().replace("/", os.sep)
    for root in roots.sorted_roots():
        candidate = root / rel
        if fs.exists(candidate):
            canon = candidate.resolve()
            canonical_hits[canon] = root

    if not canonical_hits:
        raise ModuleNotFound(
            module_id,
            roots.sorted_roots(),
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

    Globs ``<root>/<prefix>.agl`` (the prefix module itself, if it exists) and
    ``<root>/<prefix>/**/*.agl`` (the full subtree) across **all** roots.

    Each discovered file is mapped to its slash-path :class:`~agm.agl.modules.ids.ModuleId`
    via the inverse of ``ModuleId.relpath()``.  Global uniqueness is enforced: if
    the same id is found in two roots as distinct canonical files, an
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
    prefix_dir = os.sep.join(prefix)  # e.g. "foo/bar" (os-specific)

    # Accumulated results: module_id → set of distinct canonical paths found.
    # Using a set of canonicals handles dedup of symlinked/duplicate roots.
    hits: dict[ModuleId, set[Path]] = {}

    def _record_file(file_path: Path, root: Path) -> None:
        """Record a confirmed .agl file path, computing its ModuleId from root.

        *file_path* must be a file that resides under *root* and whose name
        ends with ``.agl``; these invariants are guaranteed by the caller.
        """
        canon = file_path.resolve()
        rel = file_path.relative_to(root)
        parts = rel.parts
        # rel.parts is non-empty (file_path is inside root) and last part ends
        # with ".agl" (guaranteed by the glob pattern and the name check above).
        segments = (*parts[:-1], parts[-1][:-4])  # strip .agl extension
        mid = ModuleId(segments=segments)
        if mid not in hits:
            hits[mid] = set()
        hits[mid].add(canon)

    for root in roots.sorted_roots():
        # Pattern 1: <root>/<prefix>.agl — the prefix module itself
        direct = root / (prefix_dir + ".agl")
        if fs.is_file(direct):
            _record_file(direct, root)

        # Pattern 2: <root>/<prefix>/**/*.agl — the full subtree
        subtree_root = root / prefix_dir
        if fs.is_dir(subtree_root):
            for file_path in fs.rglob(subtree_root, "*.agl"):
                if fs.is_file(file_path):
                    _record_file(file_path, root)

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
