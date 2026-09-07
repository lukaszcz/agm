"""List active AGM packages."""

from __future__ import annotations

import sys

from agm.cli_support.args import PkgListArgs
from agm.config.context import current_config_context
from agm.packages.activation import (
    CommandShadow,
    PackageActivationError,
    active_package_version,
    command_shadow_diagnostics,
    load_activation_index,
    reconcile_package_commands,
    resolve_indexed_packages,
)
from agm.packages.errors import PackageInstallError
from agm.packages.manifest import ManifestError
from agm.packages.model import canonical_package_identity
from agm.packages.store import installed_packages


def run(args: PkgListArgs) -> None:
    """Print each globally active package and its current registrations."""

    del args
    context = current_config_context()
    try:
        index = load_activation_index(home=context.home)
    except PackageActivationError as exc:
        print(f"pkg list: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    try:
        packages = installed_packages(home=context.home)
        active_packages = resolve_indexed_packages(index, home=context.home)
        index = reconcile_package_commands(index, home=context.home, packages=active_packages)
        shadows = command_shadow_diagnostics(index, home=context.home, packages=active_packages)
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
    commands_by_package: dict[str, list[str]] = {}
    for path_name, command in index.commands.items():
        commands_by_package.setdefault(command.package, []).append(path_name)
    for package in packages:
        is_active = (
            canonical_package_identity(package.manifest.name, package.manifest.version)
            in active_identities
        )
        kind = "active" if is_active else "installed"
        print(f"{package.manifest.name} {package.manifest.version} {kind}")
        if is_active:
            _print_package_commands(
                commands_by_package.get(package.manifest.name, []),
                shadows.get(package.manifest.name, ()),
            )
    for name, active in sorted(index.packages.items()):
        if active.editable is not None:
            print(f"{name} {editable_versions[name]} editable")
            _print_package_commands(commands_by_package.get(name, []), shadows.get(name, ()))


def _print_package_commands(commands: list[str], shadows: tuple[CommandShadow, ...]) -> None:
    shadowed_by_path = {shadow.path_name: shadow.displaced_packages for shadow in shadows}
    for path_name in sorted(commands):
        displaced = shadowed_by_path.get(path_name)
        if displaced is None:
            print(f"  command {path_name}")
        else:
            print(f"  command {path_name} (shadows {', '.join(displaced)})")
