"""Shared effective module-root assembly for ``agm exec`` host paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agm.agl.modules.roots import RootSet, assemble_roots
from agm.config.module_roots import (
    load_module_roots,
    resolve_lib_root,
    resolve_stdlib_root,
)
from agm.packages.activation import select_package_roots
from agm.packages.development import discover_development_packages
from agm.packages.model import PackageInfo, owning_package


@dataclass(frozen=True, slots=True)
class ExecRoots:
    """The assembled root set for an ``agm exec`` invocation.

    ``development_packages`` are the path-configured, uninstalled package
    checkouts discovered from *entry_path*/*cwd* — exposed so a caller can
    also test entry ownership against them (e.g. to route a directly executed
    development-package file to its package-qualified config route) without
    discovering them a second time.
    """

    roots: RootSet
    development_packages: tuple[PackageInfo, ...]


def effective_exec_roots(
    *,
    entry_path: Path | None,
    module_paths: list[str],
    cwd: Path,
    home: Path,
    proj_dir: Path | None,
) -> ExecRoots:
    """Build exactly the root set an ``agm exec`` invocation uses.

    Development packages reachable from *entry_path* (or *cwd*) are
    discovered here, so callers no longer need to discover and pass them in
    themselves.
    """
    development_packages = discover_development_packages(entry_path or cwd, home=home)
    module_config = load_module_roots(home=home, proj_dir=proj_dir, cwd=cwd)
    selected_packages = select_package_roots(
        home=home,
        proj_dir=proj_dir,
        cwd=cwd,
        development_packages=development_packages,
    )
    package_entry = (
        entry_path is not None and owning_package(entry_path, selected_packages) is not None
    )
    roots = assemble_roots(
        invocation_root=None
        if package_entry
        else entry_path.parent
        if entry_path is not None
        else cwd,
        stdlib_root=resolve_stdlib_root(home=home),
        lib_root=resolve_lib_root(module_config, home=home),
        configured=module_config.extra,
        cli=module_paths,
        cwd=cwd,
        package_roots=selected_packages,
    )
    return ExecRoots(roots=roots, development_packages=development_packages)
