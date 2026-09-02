"""Shared effective module-root assembly for ``agm exec`` and ``agm check``."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from agm.agl.modules.roots import RootSet, assemble_roots
from agm.config.module_roots import (
    StdlibResolutionError,
    load_module_roots,
    resolve_lib_root,
    resolve_stdlib_root,
)
from agm.packages.activation import select_package_roots
from agm.packages.development import discover_development_packages
from agm.packages.manifest import ManifestError, load_manifest
from agm.packages.model import PackageInfo, is_std_package_name, owning_package


@dataclass(frozen=True, slots=True)
class ExecRoots:
    """The assembled root set for an ``agm exec`` invocation.

    ``roots.packages`` is the mounted package selection the set was assembled
    from, so a caller can test entry ownership against exactly the packages
    that shaped the roots — a directly executed package file keeps its
    package-qualified config route whether its package is a development
    checkout or an installed store tree. The selected standard library joins
    that selection only when it owns the entry file, so every invocation that
    is not compiling a standard-library module sees the same packages it would
    have seen without the standard library's own mount.
    """

    roots: RootSet


def effective_exec_roots(
    *,
    entry_path: Path | None,
    module_paths: list[str],
    cwd: Path,
    home: Path,
    proj_dir: Path | None,
) -> ExecRoots:
    """Build exactly the root set an ``agm exec`` invocation uses.

    Development packages reachable from *entry_path* (or *cwd*) are discovered
    here and mounted alongside the selected store packages, so a caller supplies
    only the invocation's own paths.
    """
    anchor = entry_path if entry_path is not None else cwd
    development_packages = discover_development_packages(anchor, home=home)
    module_config = load_module_roots(home=home, proj_dir=proj_dir, cwd=cwd)
    selected_packages = select_package_roots(
        home=home,
        proj_dir=proj_dir,
        cwd=cwd,
        development_packages=development_packages,
    )
    stdlib_root = resolve_stdlib_root(home=home, anchor=anchor)
    stdlib_package = _mounted_stdlib_package(entry_path, stdlib_root=stdlib_root)
    mounted_packages = (
        selected_packages if stdlib_package is None else (*selected_packages, stdlib_package)
    )
    package_entry = (
        entry_path is not None and owning_package(entry_path, mounted_packages) is not None
    )
    roots = assemble_roots(
        invocation_root=None
        if package_entry
        else entry_path.parent
        if entry_path is not None
        else cwd,
        stdlib_root=stdlib_root,
        lib_root=resolve_lib_root(module_config, home=home),
        configured=module_config.extra,
        cli=module_paths,
        cwd=cwd,
        package_roots=mounted_packages,
    )
    return ExecRoots(roots=roots)


def _mounted_stdlib_package(entry_path: Path | None, *, stdlib_root: Path) -> PackageInfo | None:
    """Return the selected standard library as a package, when it owns the entry.

    ``select_package_roots`` drops every discovered ``std`` package, because
    the standard library has exactly one mounting seam. Reading the manifest at
    the root that seam already chose — a development checkout, the immutable
    store tree, the shipped tree, or an ``AGM_STDLIB`` override that happens to
    point at a real ``std`` package — gives a directly executed standard-library
    file the same package ownership any other package file has: a
    package-qualified config route and package import visibility. (``resource``
    anchoring is unaffected: it already resolves standard-library modules to the
    stdlib root.) A root without a readable ``std`` manifest, such as a synthetic
    override tree, is not a package and is not mounted.

    Mounting is scoped to an entry inside the package's module tree, so an
    invocation that is not compiling a standard-library module keeps exactly
    the package selection it would have had. The root is already the stdlib
    root, so it stays a single root either way.
    """

    if entry_path is None:
        return None
    try:
        manifest = load_manifest(stdlib_root / "package.toml")
    except ManifestError:
        return None
    if not is_std_package_name(manifest.name):
        return None
    candidate = PackageInfo(stdlib_root, manifest)
    return candidate if owning_package(entry_path, (candidate,)) is not None else None


def effective_exec_roots_or_none(
    *,
    entry_path: Path | None,
    module_paths: list[str],
    cwd: Path,
    home: Path,
    proj_dir: Path | None,
) -> ExecRoots | None:
    """Build the effective module roots, reporting an invalid configuration.

    Same parameters as :func:`effective_exec_roots`. Catches
    ``StdlibResolutionError``/``ValueError``, prints the single canonical
    ``Error: invalid module roots configuration: {exc}`` line to stderr, and
    returns ``None`` instead of raising — so every caller of this helper
    reports the same message the same way.
    """
    try:
        return effective_exec_roots(
            entry_path=entry_path,
            module_paths=module_paths,
            cwd=cwd,
            home=home,
            proj_dir=proj_dir,
        )
    except (StdlibResolutionError, ValueError) as exc:
        print(f"Error: invalid module roots configuration: {exc}", file=sys.stderr)
        return None
