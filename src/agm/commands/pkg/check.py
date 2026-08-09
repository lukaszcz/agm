"""Validate a package directory for ``agm pkg check``."""

from __future__ import annotations

import sys
from pathlib import Path

from agm.cli_support.args import PkgCheckArgs
from agm.config.context import current_config_context
from agm.packages.dependencies import DependencyError, validate_dependencies
from agm.packages.discipline import DisciplineError, validate_package
from agm.packages.manifest import ManifestError, load_manifest
from agm.packages.model import PackageInfo


def run(args: PkgCheckArgs) -> None:
    """Validate the manifest and package discipline at the selected directory."""

    root = Path.cwd() if args.directory is None else Path(args.directory)
    try:
        package = PackageInfo(root=root, manifest=load_manifest(root / "package.toml"))
        validate_package(package)
        validate_dependencies(package, home=current_config_context().home)
    except (DependencyError, DisciplineError, ManifestError) as exc:
        print(f"pkg check: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
