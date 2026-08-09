"""Create a portable archive from a checked package directory."""

from __future__ import annotations

import sys
from pathlib import Path

from agm.cli_support.args import PkgCreateArgs
from agm.config.context import current_config_context
from agm.core import dry_run
from agm.packages.archive import ArchiveError, write_archive
from agm.packages.dependencies import DependencyError, validate_dependencies
from agm.packages.discipline import DisciplineError, validate_package
from agm.packages.manifest import ManifestError, distribution_manifest, load_manifest
from agm.packages.model import PackageInfo


def run(args: PkgCreateArgs) -> None:
    """Validate a package and write its distribution archive."""

    root = Path.cwd() if args.directory is None else Path(args.directory)
    try:
        package = PackageInfo(root=root, manifest=load_manifest(root / "package.toml"))
        validate_package(package)
        context = current_config_context()
        validate_dependencies(
            PackageInfo(package.root, distribution_manifest(package.manifest)),
            home=context.home,
        )
        destination = (
            package.root.parent / f"{package.manifest.name}-{package.manifest.version}.agmpkg"
            if args.output is None
            else Path(args.output)
        )
        if dry_run.enabled():
            dry_run.print_operation("create-package-archive", str(destination))
        else:
            write_archive(package.root, destination)
    except (ArchiveError, DependencyError, DisciplineError, ManifestError) as exc:
        print(f"pkg create: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(destination)
