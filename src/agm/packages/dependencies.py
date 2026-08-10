"""Non-mutating package dependency satisfiability checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import semver

from agm.packages.install import PackageInstallError, installed_packages
from agm.packages.manifest import DependencySpec, ManifestError, load_manifest
from agm.packages.model import PackageInfo
from agm.packages.record import RecordError, verify_record
from agm.version import AGM_VERSION


class DependencyError(ValueError):
    """Raised when a package's declared dependencies cannot be satisfied."""


@dataclass(slots=True)
class _CheckState:
    home: Path
    env: Mapping[str, str] | None
    checking: set[Path] = field(default_factory=set)


def validate_dependencies(
    package: PackageInfo, *, home: Path, env: Mapping[str, str] | None = None
) -> None:
    """Ensure every requirement has a usable store, path, or URL source.

    Stored versions take precedence, matching installation's MVS resolution.
    When no stored version satisfies a requirement, a local path source is
    recursively checked. A URL with its required digest is a satisfiable
    deferred source; validation never fetches it.
    """

    _validate_package_dependencies(package, _CheckState(home=home, env=env))


def _validate_package_dependencies(package: PackageInfo, state: _CheckState) -> None:
    root = package.root.resolve()
    if root in state.checking:
        raise DependencyError(f"cyclic path dependency at {root}")
    state.checking.add(root)
    try:
        for name, requirement in package.manifest.dependencies.items():
            # The standard library is supplied by the running AGM binary
            # rather than the package store.
            if name == "std":
                if requirement.version > semver.Version.parse(AGM_VERSION):
                    raise DependencyError(
                        f"package {package.manifest.name!r} requires AGM at least "
                        f"{requirement.version} via std, but running AGM is {AGM_VERSION}"
                    )
                continue
            selected = _stored_satisfying(name, requirement, state)
            if selected is None and requirement.path is not None:
                selected = _path_package(package, name, requirement, state)
            if selected is None and requirement.url is not None:
                continue
            if selected is None or (
                selected.manifest.name != name or selected.manifest.version < requirement.version
            ):
                raise DependencyError(
                    f"unsatisfied package requirement {name!r} >= {requirement.version}"
                )
    finally:
        state.checking.remove(root)


def _stored_satisfying(
    name: str, requirement: DependencySpec, state: _CheckState
) -> PackageInfo | None:
    try:
        candidates = [
            package
            for package in installed_packages(home=state.home, env=state.env)
            if package.manifest.name == name and package.manifest.version >= requirement.version
        ]
    except PackageInstallError as exc:
        raise DependencyError(str(exc)) from exc
    if not candidates:
        return None
    selected = candidates[0]
    for candidate in candidates[1:]:
        if candidate.manifest.version > selected.manifest.version:
            selected = candidate
    try:
        verify_record(selected.root)
    except RecordError as exc:
        raise DependencyError(f"package integrity check failed for {name!r}: {exc}") from exc
    return selected


def _path_package(
    package: PackageInfo, name: str, requirement: DependencySpec, state: _CheckState
) -> PackageInfo:
    assert requirement.path is not None
    root = package.root / requirement.path
    if root.is_symlink():
        raise DependencyError(f"cannot use symbolic-link package dependency {root}")
    try:
        selected = PackageInfo(root.resolve(), load_manifest(root / "package.toml"))
    except ManifestError as exc:
        raise DependencyError(f"cannot load path dependency {name!r}: {exc}") from exc
    if selected.manifest.name != name or selected.manifest.version < requirement.version:
        raise DependencyError(f"unsatisfied package requirement {name!r} >= {requirement.version}")
    _validate_package_dependencies(selected, state)
    return selected
