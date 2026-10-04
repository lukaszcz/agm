"""Select an installed package version as the global active version."""

from __future__ import annotations

import sys

from agm.cli_support.args import PkgSwitchArgs
from agm.config.context import current_config_context
from agm.packages.install import (
    PackageInstallError,
    activate_installed_package_with_plan,
    parse_versioned_package_target,
)


def run(args: PkgSwitchArgs) -> None:
    """Activate the specified installed package version."""

    context = current_config_context()
    try:
        name, version = parse_versioned_package_target(args.target)
        activate_installed_package_with_plan(name, version, home=context.home)
    except PackageInstallError as exc:
        print(f"pkg switch: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"activated {name} {version}")
