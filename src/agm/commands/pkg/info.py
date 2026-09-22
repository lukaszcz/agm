"""Display metadata for an active AGM package."""

from __future__ import annotations

import sys
import textwrap

from agm.cli_support.args import PkgInfoArgs
from agm.config.context import current_config_context
from agm.core.env import help_width
from agm.packages.activation import (
    PackageActivationError,
    active_package_version,
    load_activation_index,
    resolve_active_package,
)
from agm.packages.manifest import ManifestError
from agm.packages.model import (
    is_std_package_name,
    std_compatibility_upper_bound,
    unmet_std_requirement,
)
from agm.util.text import first_paragraph
from agm.version import AGM_VERSION


def run(args: PkgInfoArgs) -> None:
    """Print manifest metadata and dependency status for one active package."""

    context = current_config_context()
    try:
        index = load_activation_index(home=context.home)
        active = index.packages.get(args.name)
        if active is None:
            raise PackageActivationError(f"package {args.name!r} is not installed")
        manifest = resolve_active_package(args.name, active, home=context.home).manifest
        active_versions = {
            name: active_package_version(selected) for name, selected in index.packages.items()
        }
    except (ManifestError, PackageActivationError) as exc:
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
    if manifest.aliases:
        print("aliases:")
        for alias, target in sorted(manifest.aliases.items()):
            print(f"  {alias}: {target}")
    if manifest.commands:
        print("commands:")
        width = help_width()
        for path, command in sorted(manifest.commands.items()):
            print(f"  {path}: {command.program or 'command group'}")
            if command.doc:
                print(
                    textwrap.fill(
                        first_paragraph(command.doc),
                        width=width,
                        initial_indent="      ",
                        subsequent_indent="      ",
                    )
                )
    for name, dependency in sorted(manifest.dependencies.items()):
        requirement = f">= {dependency.version}"
        if is_std_package_name(name):
            requirement += f", < {std_compatibility_upper_bound(dependency)}"
            status = f"running AGM {AGM_VERSION}"
            if unmet_std_requirement(dependency) is not None:
                status += " (unsatisfied)"
        else:
            selected = index.packages.get(name)
            if selected is None:
                status = "missing"
            elif active_versions[name] < dependency.version:
                status = f"active {active_versions[name]} (unsatisfied)"
            else:
                kind = "editable" if selected.editable is not None else "active"
                status = f"{kind} {active_versions[name]}"
        print(f"requires {name} {requirement}: {status}")
