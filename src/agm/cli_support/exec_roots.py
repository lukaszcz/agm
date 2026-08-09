"""Shared effective module-root assembly for ``agm exec`` host paths."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from agm.agl.modules.roots import RootSet, assemble_roots

if TYPE_CHECKING:
    from agm.packages.model import PackageInfo
from agm.config.module_roots import (
    load_module_roots,
    resolve_lib_root,
    resolve_stdlib_root,
)


def effective_exec_roots(
    *,
    entry_path: Path | None,
    module_paths: list[str],
    cwd: Path,
    home: Path,
    proj_dir: Path | None,
    package_roots: Iterable[PackageInfo] = (),
) -> RootSet:
    """Build exactly the root set an ``agm exec`` invocation uses."""
    module_config = load_module_roots(home=home, proj_dir=proj_dir, cwd=cwd)
    return assemble_roots(
        invocation_root=entry_path.parent if entry_path is not None else cwd,
        stdlib_root=resolve_stdlib_root(home=home),
        lib_root=resolve_lib_root(module_config, home=home),
        configured=module_config.extra,
        cli=module_paths,
        cwd=cwd,
        package_roots=package_roots,
    )
