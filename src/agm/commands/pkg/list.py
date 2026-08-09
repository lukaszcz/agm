"""List active AGM packages."""

from __future__ import annotations

import sys

from agm.cli_support.args import PkgListArgs
from agm.config.context import current_config_context
from agm.packages.activation import PackageActivationError, load_activation_index
from agm.packages.install import PackageInstallError, installed_packages


def run(args: PkgListArgs) -> None:
    """Print each globally active package and its activation kind."""

    del args
    context = current_config_context()
    try:
        index = load_activation_index(home=context.home)
    except PackageActivationError as exc:
        print(f"pkg list: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    try:
        packages = installed_packages(home=context.home)
    except PackageInstallError as exc:
        print(f"pkg list: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    active_versions = {
        (name, active.version) for name, active in index.packages.items() if active.editable is None
    }
    for package in packages:
        kind = (
            "active"
            if (package.manifest.name, package.manifest.version) in active_versions
            else "installed"
        )
        print(f"{package.manifest.name} {package.manifest.version} {kind}")
    for name, active in sorted(index.packages.items()):
        if active.editable is not None:
            print(f"{name} {active.version} editable")
