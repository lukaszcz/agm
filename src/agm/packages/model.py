"""Package identity and canonical module-file ownership."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import semver

from agm.packages.layout import MODULE_TREE_DIRNAME
from agm.packages.manifest import DependencySpec, PackageManifest
from agm.version import AGM_VERSION

PackageIdentity = tuple[str, str]

STD_PACKAGE_NAME = "std"
"""The name of AGM's managed, lockstep standard-library package."""


def canonical_package_identity(name: str, version: semver.Version) -> PackageIdentity:
    """Return an exact canonical identity, including semantic-version build metadata."""

    return name, str(version)


def is_std_package_name(name: str) -> bool:
    """Return whether *name* names AGM's managed, lockstep standard-library package."""

    return name == STD_PACKAGE_NAME


def std_compatibility_upper_bound(requirement: DependencySpec) -> semver.Version:
    """Return the exclusive AGM compatibility bound for a ``std`` requirement.

    Pre-1.0 releases are compatible within one minor line; stable releases
    are compatible within one major line.
    """

    version = requirement.version
    if version.major == 0:
        return semver.Version(0, version.minor + 1, 0)
    return semver.Version(version.major + 1, 0, 0)


def unmet_std_requirement(requirement: DependencySpec) -> str | None:
    """Describe an incompatible ``std`` dependency, or return ``None``.

    ``std`` is shipped by AGM rather than resolved like an ordinary package.
    Its version is a minimum within one compatible AGM release line. Callers
    prefix the returned clause with their own subject and raise their own
    error type.
    """

    running = semver.Version.parse(AGM_VERSION)
    upper_bound = std_compatibility_upper_bound(requirement)
    if requirement.version <= running < upper_bound:
        return None
    return (
        f"requires AGM >= {requirement.version}, < {upper_bound} via std, "
        f"but running AGM is {AGM_VERSION}"
    )


@dataclass(frozen=True, slots=True)
class PackageInfo:
    """A manifest paired with its canonical package-root directory."""

    root: Path
    manifest: PackageManifest

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.resolve())

    @property
    def module_root(self) -> Path:
        """Return the package's module-tree directory."""

        return (self.root / MODULE_TREE_DIRNAME).resolve()

    def module_path(self, segments: Sequence[str]) -> Path:
        """Return the file this package's module tree gives a module id.

        A module id names the package in its leading segment and the file
        beneath the module tree in the rest, so a bare package name names no
        module. Raises :class:`ValueError` for segments this package does not
        name a module for.
        """

        if len(segments) < 2 or segments[0] != self.manifest.name:
            raise ValueError(
                f"{'/'.join(segments)!r} is not a module of package {self.manifest.name!r}"
            )
        return self.module_root.joinpath(*segments[1:-1], f"{segments[-1]}.agl")

    def module_id_segments(self, path: Path) -> tuple[str, ...]:
        """Return the module-id segments this package gives a file in its module tree.

        The inverse of :meth:`module_path`. Raises :class:`ValueError` for a
        path outside the module tree.
        """

        relative = path.resolve().relative_to(self.module_root).with_suffix("")
        return (self.manifest.name, *relative.parts)


class VersionSelection(Protocol):
    """The version/editable shape ``select_satisfying`` needs from an active selection.

    A structural protocol lets this module accept ``activation.ActivePackage``
    (and any similar store-domain caller) without importing it, since
    ``activation`` itself imports this module. Read-only properties (rather
    than plain attributes) match a frozen dataclass's read-only fields.
    """

    @property
    def version(self) -> semver.Version: ...

    @property
    def editable(self) -> Path | None: ...


def select_satisfying(
    candidates: Sequence[PackageInfo], active: VersionSelection | None
) -> PackageInfo | None:
    """Return the greatest satisfying candidate, preferring the active build on a tie.

    Versions that compare equal can still differ in build metadata, so an
    exact match for the current selection wins over an equivalent sibling.
    This is the shared MVS choice made by installation, dependency
    validation, and activation-index rebuilding.
    """

    if not candidates:
        return None
    selected = candidates[0]
    for candidate in candidates[1:]:
        if candidate.manifest.version > selected.manifest.version or (
            candidate.manifest.version == selected.manifest.version
            and active is not None
            and active.editable is None
            and str(candidate.manifest.version) == str(active.version)
        ):
            selected = candidate
    return selected


def owning_package(path: Path, packages: tuple[PackageInfo, ...]) -> PackageInfo | None:
    """Return the package whose module tree canonically contains *path*.

    Assets and manifests are deliberately not owned by this map: ownership is
    the module-loader policy seam, so only files under a module tree qualify.
    """

    canonical_path = path.resolve()
    for package in sorted(packages, key=_module_root_length, reverse=True):
        try:
            canonical_path.relative_to(package.module_root)
        except ValueError:
            continue
        return package
    return None


def _module_root_length(package: PackageInfo) -> int:
    """Order nested module roots before their containing roots."""
    return len(str(package.module_root))
