"""Non-mutating package dependency satisfiability checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from agm.packages.activation import (
    ActivationIndex,
    PackageActivationError,
    load_activation_index,
)
from agm.packages.discipline import DisciplineError, validate_package_structure
from agm.packages.install import PackageInstallError, installed_packages
from agm.packages.manifest import DependencySpec, ManifestError, load_manifest
from agm.packages.model import PackageInfo, is_std_package_name, unmet_std_requirement
from agm.packages.store import satisfying_from_store


class DependencyError(ValueError):
    """Raised when a package's declared dependencies cannot be satisfied."""


@dataclass(slots=True)
class _CheckState:
    home: Path
    env: Mapping[str, str] | None
    checking: set[Path] = field(default_factory=set)
    checked: set[Path] = field(default_factory=set)
    resolved: dict[str, PackageInfo] = field(default_factory=dict)
    installed: tuple[PackageInfo, ...] | None = None
    index: ActivationIndex | None = None


def validate_dependencies(
    package: PackageInfo, *, home: Path, env: Mapping[str, str] | None = None
) -> tuple[PackageInfo, ...]:
    """Ensure every requirement has a usable store, path, or URL source.

    Stored and previously selected versions take precedence, matching
    installation's MVS resolution. When no selected version satisfies a
    requirement, a local path source is recursively checked. A URL with its
    required digest is a satisfiable
    deferred source; validation never fetches it.
    """

    state = _CheckState(home=home, env=env)
    _validate_package_dependencies(package, state)
    return tuple(state.resolved[name] for name in sorted(state.resolved))


def _validate_package_dependencies(package: PackageInfo, state: _CheckState) -> None:
    # A package reached again through a second path resolves identically, so
    # revisiting it would only re-walk its subtree exponentially.
    root = package.root.resolve()
    if root in state.checking:
        raise DependencyError(f"cyclic path dependency at {root}")
    if root in state.checked:
        return
    state.checking.add(root)
    try:
        for name, requirement in package.manifest.dependencies.items():
            # The standard library is supplied by the running AGM binary
            # rather than the package store.
            if is_std_package_name(name):
                unmet = unmet_std_requirement(requirement)
                if unmet is not None:
                    raise DependencyError(f"package {package.manifest.name!r} {unmet}")
                continue
            selected = _selected_satisfying(name, requirement, state)
            if selected is not None:
                _validate_package_dependencies(selected, state)
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
            state.resolved[selected.manifest.name] = selected
    finally:
        state.checking.remove(root)
    state.checked.add(root)


def _cached_installed_packages(state: _CheckState) -> tuple[PackageInfo, ...]:
    """Return the store contents, scanned once per dependency check."""

    if state.installed is None:
        state.installed = installed_packages(home=state.home, env=state.env)
    return state.installed


def _cached_activation_index(state: _CheckState) -> ActivationIndex:
    """Return the activation index, loaded once per dependency check."""

    if state.index is None:
        state.index = load_activation_index(home=state.home, env=state.env)
    return state.index


def _selected_satisfying(
    name: str, requirement: DependencySpec, state: _CheckState
) -> PackageInfo | None:
    # A name already resolved in this walk stays a candidate: it may have come
    # from a path source that is not in the store, and a diamond must select
    # one package per name across every route to it.
    previously_selected = state.resolved.get(name)
    extra = () if previously_selected is None else (previously_selected,)
    try:
        installed = _cached_installed_packages(state)
        active = _cached_activation_index(state).packages.get(name)
        return satisfying_from_store(installed, name, requirement, active, extra_candidates=extra)
    except (PackageActivationError, PackageInstallError) as exc:
        raise DependencyError(str(exc)) from exc


def _path_package(
    package: PackageInfo, name: str, requirement: DependencySpec, state: _CheckState
) -> PackageInfo:
    assert requirement.path is not None
    root = package.root / requirement.path
    if root.is_symlink():
        raise DependencyError(f"cannot use symbolic-link package dependency {root}")
    try:
        selected = PackageInfo(root.resolve(), load_manifest(root / "package.toml"))
        validate_package_structure(selected)
    except (DisciplineError, ManifestError) as exc:
        raise DependencyError(f"cannot validate path dependency {name!r}: {exc}") from exc
    if selected.manifest.name != name or selected.manifest.version < requirement.version:
        raise DependencyError(f"unsatisfied package requirement {name!r} >= {requirement.version}")
    _validate_package_dependencies(selected, state)
    return selected
