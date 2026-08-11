"""Root-set types and assembly for the AgL module system."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agm.packages.model import PackageInfo


@dataclass(frozen=True, slots=True)
class RootSet:
    """An unordered, de-duplicated, canonical set of module-search roots.

    All paths stored in :attr:`roots` are absolute and canonical
    (``Path.resolve()`` applied).  The set is unordered by design — an AgL
    module id must resolve to *at most one* file across all roots (ambiguity
    is an error).

    Package-only roots are scoped to their declared top-level module segment.
    :attr:`loose_roots` records ordinary roots that overlap package roots, so
    their unrestricted semantics win. Use :meth:`sorted_roots` for deterministic
    output in diagnostics and :meth:`sorted_roots_for` for resolution.
    """

    roots: frozenset[Path]
    packages: tuple[PackageInfo, ...] = ()
    stdlib_roots: frozenset[Path] = frozenset()
    loose_roots: frozenset[Path] = frozenset()

    def sorted_roots(self) -> tuple[Path, ...]:
        """Return roots sorted lexicographically for deterministic diagnostics."""
        return tuple(sorted(self.roots))

    def sorted_roots_for(self, prefix: tuple[str, ...]) -> tuple[Path, ...]:
        """Return roots whose mount scope admits a module *prefix*.

        Ordinary roots are loose. A root supplied only by a package mount
        admits that package's declared top-level module segment; explicitly
        supplying the same path as an ordinary root keeps it loose.
        """
        package_names_by_root: dict[Path, set[str]] = {}
        for package in self.packages:
            package_names_by_root.setdefault(package.root, set()).add(package.manifest.name)
        first_segment = prefix[0]
        return tuple(
            root
            for root in self.sorted_roots()
            if root in self.loose_roots
            or root not in package_names_by_root
            or first_segment in package_names_by_root[root]
        )

    def admits_path(self, root: Path, path: Path) -> bool:
        """Return whether *path* is exposed by *root*'s mount policy."""
        if root in self.loose_roots:
            return True
        mounted = tuple(package for package in self.packages if package.root == root)
        if not mounted:
            return True
        canonical_path = path.resolve()
        return any(canonical_path.is_relative_to(package.module_root) for package in mounted)

    def is_standard_library_path(self, path: Path) -> bool:
        """Return whether *path* belongs to a host-selected standard-library root."""
        canonical_path = path.resolve()
        return any(canonical_path.is_relative_to(stdlib_root) for stdlib_root in self.stdlib_roots)


def _canonicalize(path: Path) -> Path:
    """Expand user (~), make absolute, and resolve symlinks/relative components."""
    expanded = Path(os.path.expanduser(str(path)))
    return expanded.resolve()


def assemble_roots(
    *,
    invocation_root: Path | None,
    stdlib_root: Path | None = None,
    lib_root: Path | None,
    configured: Iterable[tuple[str, Path]],
    cli: Iterable[str],
    cwd: Path,
    package_roots: Iterable[PackageInfo] = (),
) -> RootSet:
    """Assemble a :class:`RootSet` from all root sources.

    Parameters
    ----------
    invocation_root:
        The cwd (for ``exec -c``) or a loose entry file's directory. ``None``
        keeps a package-owned entry's directory from becoming a loose root.
    stdlib_root:
        The selected standard-library module root (normally the active
        ``<AGM home>/packages/std/<AGM version>`` tree), or ``None`` if the
        caller does not want to add one.
    lib_root:
        The global library root (e.g. ``~/.agm/lib``), or ``None`` if not
        configured.  Applied as-is; caller supplies the default if desired.
    configured:
        ``(raw_path, origin_dir)`` pairs from AGM config, already interpolated
        by :func:`agm.config.module_roots.load_module_roots`.  Relative
        *raw_path* values are resolved against *origin_dir* (the directory of
        the config file that declared them).
    cli:
        Raw path strings from the ``-I``/``--module-path`` CLI flag.  Relative
        paths are resolved against *cwd*.
    cwd:
        Current working directory; used to resolve relative CLI paths.
    package_roots:
        Mounted non-``std`` packages, regardless of whether their roots come
        from a development directory or a future package store. A supplied
        ``std`` package is ignored because ``stdlib_root`` is its exclusive
        mounting seam. Package mounts expose only the package's declared
        module tree unless the same path is also supplied as an ordinary root.

    All roots are user-expanded, made absolute, and canonicalized before
    de-duplication.  Non-existent roots are dropped silently (resolution
    errors are reported later by the resolver, which lists the searched set).
    """
    canonical_roots: set[Path] = set()
    loose_roots: set[Path] = set()

    def _add(path: Path) -> Path | None:
        canonical_path = _canonicalize(path)
        if canonical_path.exists():
            canonical_roots.add(canonical_path)
            loose_roots.add(canonical_path)
            return canonical_path
        return None

    # 1. Invocation root
    if invocation_root is not None:
        _add(invocation_root)

    # 2. Standard library root. Keep its distinct identity so package
    # visibility can admit host-provided modules without admitting loose roots.
    stdlib_roots: set[Path] = set()
    if stdlib_root is not None:
        canonical_stdlib_root = _add(stdlib_root)
        if canonical_stdlib_root is not None:
            stdlib_roots.add(canonical_stdlib_root)

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

    # 6. Package roots. Keep the package metadata only when its root is
    # mounted, so ownership policy and resolver see the same selection.
    mounted_packages: list[PackageInfo] = []
    for package in package_roots:
        # ``std`` is mounted only through ``stdlib_root``. This keeps an
        # override or source-checkout fallback exclusive of active packages.
        if package.manifest.name == "std":
            continue
        package_root = _canonicalize(package.root)
        if package_root.exists():
            canonical_roots.add(package_root)
            mounted_packages.append(package)

    return RootSet(
        roots=frozenset(canonical_roots),
        packages=tuple(mounted_packages),
        stdlib_roots=frozenset(stdlib_roots),
        loose_roots=frozenset(loose_roots),
    )
