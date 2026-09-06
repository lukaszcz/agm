"""Root-set types and assembly for the AgL module system."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from agm.agl.modules.ids import ModuleId
from agm.packages.layout import MODULE_TREE_DIRNAME
from agm.packages.model import STD_PACKAGE_NAME, PackageInfo, owning_package


@dataclass(frozen=True, slots=True)
class RootSet:
    """The module-search surface of one invocation: loose roots plus mounts.

    A *loose* root in :attr:`roots` resolves any id as ``<root>/<id>.agl``.  A
    *mount* binds one top-level id segment to a module-tree directory, so
    ``<name>/<rest>`` resolves as ``<tree>/<rest>.agl`` and nothing else under
    the mounted package's root is a module unless the same directory is also
    supplied as a loose root.  Each package in :attr:`packages` mounts its
    declared name, and each standard-library root in :attr:`stdlib_roots`
    mounts ``std``; a library root that is also mounted as a package
    contributes the same tree once.

    All paths stored here are absolute and canonical (``Path.resolve()``
    applied).  The set is unordered by design — an AgL module id must resolve
    to *at most one* file across every loose root and mount (ambiguity is an
    error).  Use :meth:`sorted_roots` for deterministic output in diagnostics
    and :meth:`module_tree_roots` for the mounts a resolution consults.

    The mount table is derived once here in ``__post_init__``, since a module
    id is resolved against the same :class:`RootSet` many times (once per
    import edge, and once per matched file during wildcard expansion).
    """

    roots: frozenset[Path]
    packages: tuple[PackageInfo, ...] = ()
    stdlib_roots: frozenset[Path] = frozenset()
    _sorted_roots: tuple[Path, ...] = field(init=False, repr=False, compare=False)
    _module_trees: dict[str, tuple[Path, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        trees: dict[str, set[Path]] = {}
        for package in self.packages:
            trees.setdefault(package.manifest.name, set()).add(package.module_root)
        for stdlib_root in self.stdlib_roots:
            trees.setdefault(STD_PACKAGE_NAME, set()).add(
                (stdlib_root / MODULE_TREE_DIRNAME).resolve()
            )
        sorted_roots: tuple[Path, ...] = tuple(sorted(self.roots))
        module_trees: dict[str, tuple[Path, ...]] = {
            name: tuple(sorted(paths)) for name, paths in trees.items()
        }
        object.__setattr__(self, "_sorted_roots", sorted_roots)
        object.__setattr__(self, "_module_trees", module_trees)

    def sorted_roots(self) -> tuple[Path, ...]:
        """Return the loose roots sorted lexicographically for deterministic diagnostics."""
        return self._sorted_roots

    def module_tree_roots(self, name: str) -> tuple[Path, ...]:
        """Return the module-tree directories mounted for a top-level id segment.

        Sorted and deduplicated by canonical path, so a library root mounted
        both as the standard library and as its own ``std`` package is searched
        once.
        """
        return self._module_trees.get(name, ())

    def package_module_id_for(self, path: Path) -> ModuleId | None:
        """Return the module id a mounted package gives *path*, if one owns it.

        A package manifest declares a module tree, so a file inside one has a
        module identity independent of how it was reached: the loader keys a
        directly executed or checked package file by this id rather than by the
        anonymous entry sentinel, and the command layer routes its
        configuration under the same path.  ``None`` for a file no mounted
        package owns — a loose root is where the user happened to invoke the
        tool, not a declaration that its files are modules.
        """
        package = owning_package(path, self.packages)
        if package is None:
            return None
        return ModuleId(segments=package.module_id_segments(path))

    def is_standard_library_path(self, path: Path) -> bool:
        """Return whether *path* belongs to a host-selected standard-library root.

        Asked only of an import out of a package, which the standard library is
        exempt from declaring as a dependency.
        """
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
        The selected standard-library package root (normally the active
        ``<AGM home>/packages/std/<AGM version>`` tree), or ``None`` if the
        caller does not want to add one.  It mounts ``std`` rather than
        becoming a loose root, so only its module tree is importable.
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
        Packages to mount, regardless of whether their roots come from a
        development directory or a package store.  A mount exposes only the
        package's module tree; the root itself becomes a loose root only when
        some other source supplies the same path.  A ``std`` package belongs
        here only when *stdlib_root* already selected that very tree, so the
        standard library keeps exactly one mount while still owning the files
        under it.

    Loose roots are user-expanded, made absolute, and canonicalized before
    de-duplication.  Non-existent roots and packages are dropped silently
    (resolution errors are reported later by the resolver, which lists the
    searched set).
    """
    canonical_roots: set[Path] = set()

    def _add(path: Path) -> None:
        canonical_path = _canonicalize(path)
        if canonical_path.exists():
            canonical_roots.add(canonical_path)

    # 1. Invocation root
    if invocation_root is not None:
        _add(invocation_root)

    # 2. Global library root
    if lib_root is not None:
        _add(lib_root)

    # 3. Configured roots — relative paths resolve against their origin dir
    for raw, origin_dir in configured:
        raw_path = Path(os.path.expanduser(raw))
        if raw_path.is_absolute():
            _add(raw_path)
        else:
            _add(origin_dir / raw_path)

    # 4. CLI roots — relative paths resolve against cwd
    for raw in cli:
        raw_path = Path(os.path.expanduser(raw))
        if raw_path.is_absolute():
            _add(raw_path)
        else:
            _add(cwd / raw_path)

    # 5. Standard-library root. Keep its distinct identity so package
    # visibility can admit host-provided modules without admitting loose roots.
    stdlib_roots: set[Path] = set()
    if stdlib_root is not None:
        canonical_stdlib_root = _canonicalize(stdlib_root)
        if canonical_stdlib_root.exists():
            stdlib_roots.add(canonical_stdlib_root)

    # 6. Package mounts. Keep the package metadata only when its root exists,
    # so ownership policy and resolver see the same selection.
    mounted_packages = tuple(
        package for package in package_roots if _canonicalize(package.root).exists()
    )

    return RootSet(
        roots=frozenset(canonical_roots),
        packages=mounted_packages,
        stdlib_roots=frozenset(stdlib_roots),
    )
