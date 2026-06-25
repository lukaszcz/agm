"""Root-set types and assembly for the AgL module system."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RootSet:
    """An unordered, de-duplicated, canonical set of module-search roots.

    All paths stored in :attr:`roots` are absolute and canonical
    (``Path.resolve()`` applied).  The set is unordered by design — an AgL
    module id must resolve to *at most one* file across all roots (ambiguity
    is an error).

    Use :meth:`sorted_roots` for deterministic output in diagnostics.
    """

    roots: frozenset[Path]

    def sorted_roots(self) -> tuple[Path, ...]:
        """Return roots sorted lexicographically for deterministic diagnostics."""
        return tuple(sorted(self.roots))


def _canonicalize(path: Path) -> Path:
    """Expand user (~), make absolute, and resolve symlinks/relative components."""
    expanded = Path(os.path.expanduser(str(path)))
    return expanded.resolve()


def assemble_roots(
    *,
    invocation_root: Path,
    stdlib_root: Path | None = None,
    lib_root: Path | None,
    configured: Iterable[tuple[str, Path]],
    cli: Iterable[str],
    cwd: Path,
) -> RootSet:
    """Assemble a :class:`RootSet` from all root sources.

    Parameters
    ----------
    invocation_root:
        The cwd (for ``exec -c``) or the entry file's directory (for
        ``exec <file>``).
    stdlib_root:
        The selected standard-library module root (e.g. ``~/.agm/stdlib``), or
        ``None`` if the caller does not want to add one.
    lib_root:
        The global library root (e.g. ``~/.agm/lib``), or ``None`` if not
        configured.  Applied as-is; caller supplies the default if desired.
    configured:
        ``(raw_path, origin_dir)`` pairs from AGM config.  Relative *raw_path*
        values are resolved against *origin_dir* (the directory of the config
        file that declared them).
    cli:
        Raw path strings from the ``-I``/``--module-path`` CLI flag.  Relative
        paths are resolved against *cwd*.
    cwd:
        Current working directory; used to resolve relative CLI paths.

    All roots are user-expanded, made absolute, and canonicalized before
    de-duplication.  Non-existent roots are dropped silently (resolution
    errors are reported later by the resolver, which lists the searched set).
    """
    canonical_roots: set[Path] = set()

    def _add(path: Path) -> None:
        canon = _canonicalize(path)
        if canon.exists():
            canonical_roots.add(canon)

    # 1. Invocation root
    _add(invocation_root)

    # 2. Standard library root
    if stdlib_root is not None:
        _add(stdlib_root)

    # 3. Global library root
    if lib_root is not None:
        _add(lib_root)

    # 4. Configured roots — relative paths resolve against their origin dir
    for raw, origin_dir in configured:
        raw_path = Path(os.path.expanduser(raw))
        if raw_path.is_absolute():
            _add(raw_path)
        else:
            _add(origin_dir / raw_path)

    # 5. CLI roots — relative paths resolve against cwd
    for raw in cli:
        raw_path = Path(os.path.expanduser(raw))
        if raw_path.is_absolute():
            _add(raw_path)
        else:
            _add(cwd / raw_path)

    return RootSet(roots=frozenset(canonical_roots))
