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
from agm.packages.model import owning_package


@dataclass(frozen=True, slots=True)
class ExecRoots:
    """The assembled root set for an ``agm exec`` invocation.

    ``roots.packages`` is the mounted package selection the set was assembled
    from, so a caller can test entry ownership against exactly the packages
    that shaped the roots — a directly executed package file keeps its
    package-qualified config route whether its package is a development
    checkout or an installed store tree.
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
    return ExecRoots(roots=roots)


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
