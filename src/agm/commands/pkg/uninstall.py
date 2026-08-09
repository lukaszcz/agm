"""Remove an active package from AGM's store."""

from __future__ import annotations

import sys

from agm.cli_support.args import PkgUninstallArgs
from agm.config.context import current_config_context
from agm.packages.install import PackageInstallError, uninstall_package


def run(args: PkgUninstallArgs) -> None:
    """Uninstall the named active package."""

    context = current_config_context()
    try:
        uninstall_package(args.name, home=context.home)
    except PackageInstallError as exc:
        print(f"pkg uninstall: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"uninstalled {args.name}")
