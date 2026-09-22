"""List active AGM packages."""

from __future__ import annotations

import sys

from agm.cli_support.args import PkgListArgs
from agm.config.context import current_config_context
from agm.packages.activation import (
    PackageActivationError,
    active_package_version,
    load_activation_index,
)
from agm.packages.errors import PackageInstallError
from agm.packages.manifest import ManifestError
from agm.packages.model import canonical_package_identity
from agm.packages.store import installed_packages


def run(args: PkgListArgs) -> None:
    """Print every installed package version and every active editable package."""

    del args
    context = current_config_context()
    try:
        index = load_activation_index(home=context.home)
        packages = installed_packages(home=context.home)
        editable_versions = {
            name: active_package_version(active)
            for name, active in index.packages.items()
            if active.editable is not None
        }
    except (ManifestError, PackageActivationError, PackageInstallError) as exc:
        print(f"pkg list: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    active_identities = {
        canonical_package_identity(name, active.version)
        for name, active in index.packages.items()
        if active.editable is None
    }
    for package in packages:
        identity = canonical_package_identity(package.manifest.name, package.manifest.version)
        kind = "active" if identity in active_identities else "installed"
        print(f"{package.manifest.name} {package.manifest.version} {kind}")
    for name, active in sorted(index.packages.items()):
        if active.editable is not None:
            print(f"{name} {editable_versions[name]} editable")
