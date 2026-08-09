"""Display metadata for an active AGM package."""

from __future__ import annotations

import sys

from agm.cli_support.args import PkgInfoArgs
from agm.config.context import current_config_context
from agm.packages.activation import PackageActivationError, load_activation_index
from agm.packages.manifest import ManifestError, load_manifest
from agm.packages.store import StorePathError, canonical_package_store_path


def run(args: PkgInfoArgs) -> None:
    """Print manifest metadata and dependency status for one active package."""

    context = current_config_context()
    try:
        index = load_activation_index(home=context.home)
        active = index.packages.get(args.name)
        if active is None:
            raise PackageActivationError(f"package {args.name!r} is not installed")
        if active.editable is not None:
            root = active.editable
        else:
            root = canonical_package_store_path(args.name, active.version, home=context.home)
        manifest = load_manifest(root / "package.toml")
        if manifest.name != args.name:
            raise PackageActivationError(
                f"active package {args.name!r} has a manifest for {manifest.name!r}"
            )
        if active.editable is None and manifest.version != active.version:
            raise PackageActivationError(
                f"active package {args.name!r} has a mismatched installed version"
            )
    except (ManifestError, PackageActivationError, StorePathError) as exc:
        print(f"pkg info: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"{manifest.name} {manifest.version}")
    if manifest.description is not None:
        print(manifest.description)
    if manifest.license is not None:
        print(f"license: {manifest.license}")
    if manifest.authors:
        print(f"authors: {', '.join(manifest.authors)}")
    if manifest.repository is not None:
        print(f"repository: {manifest.repository}")
    if manifest.keywords:
        print(f"keywords: {', '.join(manifest.keywords)}")
    if manifest.commands:
        print("commands:")
        for path, command in sorted(manifest.commands.items()):
            description = "" if command.description is None else f" ({command.description})"
            print(f"  {path}: {command.program}{description}")
    for name, dependency in sorted(manifest.dependencies.items()):
        selected = index.packages.get(name)
        if selected is None:
            status = "missing"
        elif selected.version < dependency.version:
            status = f"active {selected.version} (unsatisfied)"
        else:
            kind = "editable" if selected.editable is not None else "active"
            status = f"{kind} {selected.version}"
        print(f"requires {name} >= {dependency.version}: {status}")
