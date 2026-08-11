"""Install a directory package into AGM's versioned store."""

from __future__ import annotations

import sys
from pathlib import Path

from agm.cli_support.args import PkgInstallArgs
from agm.config.context import current_config_context
from agm.packages.install import (
    PackageInstallError,
    install_archive_with_plan,
    install_directory_with_plan,
)


def run(args: PkgInstallArgs) -> None:
    """Install or activate the selected package directory."""

    context = current_config_context()
    try:
        source = Path(args.source)
        if source.is_file() and not args.editable:
            plan = install_archive_with_plan(
                source,
                home=context.home,
                shadow=args.shadow,
            )
        else:
            plan = install_directory_with_plan(
                source,
                home=context.home,
                editable=args.editable,
                shadow=args.shadow,
            )
    except PackageInstallError as exc:
        print(f"pkg install: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    package = plan.package
    kind = "editable" if args.editable else "installed"
    print(f"{kind} {package.manifest.name} {package.manifest.version}")
    if args.shadow:
        if plan.command_shadows:
            for shadow in plan.command_shadows:
                owners = ", ".join(shadow.displaced_packages)
                print(f"shadowed command {shadow.path_name} from {owners}")
        else:
            print("shadow requested; no active package commands were displaced")
