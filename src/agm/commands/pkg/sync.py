"""Sync AGM's interpreter environment to the active packages' Python requirements."""

from __future__ import annotations

import sys

from agm.cli_support.args import PkgSyncArgs
from agm.config.context import current_config_context
from agm.core import dry_run
from agm.packages.install import PackageInstallError, sync_active_python_dependencies


def run(args: PkgSyncArgs) -> None:
    """Install the active packages' Python requirements when any is unsatisfied."""

    del args
    try:
        installed = sync_active_python_dependencies(home=current_config_context().home)
    except PackageInstallError as exc:
        print(f"pkg sync: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if not installed:
        print("python requirements satisfied")
    verb = "would install" if dry_run.enabled() else "installed"
    for spec in installed:
        print(f"{verb} python requirement {spec}")
