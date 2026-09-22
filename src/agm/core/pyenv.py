"""Python requirements of AGM's own interpreter environment.

Package companions run in the interpreter running AGM, so their third-party
requirements are checked against, and installed into, ``sys.executable``'s
environment.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from importlib import metadata

from packaging.markers import (
    EvaluateContext,
    Marker,
    UndefinedComparison,
    UndefinedEnvironmentName,
)
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from agm.core import process


class RequirementInstallError(RuntimeError):
    """Raised when the installer cannot start or exits unsuccessfully."""


def install_requirements(specs: Sequence[str]) -> None:
    """Install PEP 508 *specs* with ``uv`` when on ``PATH``, else ``pip``; honours dry-run."""

    if not specs:
        return
    uv = shutil.which("uv")
    installer = (
        [uv, "pip", "install", "--python", sys.executable]
        if uv is not None
        else [sys.executable, "-m", "pip", "install"]
    )
    cmd = [*installer, *specs]
    hint = "" if uv is not None else f" (installing needs uv on PATH or pip in {sys.executable})"
    try:
        returncode = process.run_foreground_unless_dry_run(cmd)
    except OSError as exc:
        raise RequirementInstallError(f"cannot run {cmd[0]}: {exc}{hint}") from exc
    if returncode != 0:
        raise RequirementInstallError(f"{cmd[0]} exited with status {returncode}{hint}")


def marker_value(
    marker: Marker,
    environment: dict[str, str] | None = None,
    *,
    context: EvaluateContext = "metadata",
) -> bool | None:
    """*marker*'s value; ``None`` when it names an undefined variable or compares incomparably."""

    try:
        return marker.evaluate(environment, context=context)
    except (KeyError, UndefinedComparison, UndefinedEnvironmentName):
        return None


class RequirementState(Enum):
    """How a requirement stands in the running interpreter's environment."""

    INSTALLED = "installed"
    UNSATISFIED = "unsatisfied"
    MISSING = "missing"
    NOT_APPLICABLE = "not applicable"


@dataclass(frozen=True, slots=True)
class RequirementStatus:
    """A requirement's state and its distribution's installed version, if any."""

    state: RequirementState
    version: str | None = None

    @property
    def satisfied(self) -> bool:
        return self.state in (RequirementState.INSTALLED, RequirementState.NOT_APPLICABLE)


def requirement_status(spec: str) -> RequirementStatus:
    """Status of a manifest's PEP 508 *spec* in the running interpreter's environment.

    A marker is evaluated as a requirement's (no ``extra``). An installed
    pre-release matches when the specifier admits it, as an installer treats it.
    Each requested extra holds when every requirement the distribution declares
    only for it is satisfied, extras included.
    """

    requirement = Requirement(spec)
    marker = requirement.marker
    if marker is not None and not marker.evaluate(context="requirement"):
        return RequirementStatus(RequirementState.NOT_APPLICABLE)
    installed = _installed_version(requirement.name)
    if installed is None:
        return RequirementStatus(RequirementState.MISSING)
    holds = _installed_holds(requirement, installed, set())
    return RequirementStatus(
        RequirementState.INSTALLED if holds else RequirementState.UNSATISFIED, installed
    )


def _holds(requirement: Requirement, seen: set[tuple[str, str]]) -> bool:
    """Whether *requirement*'s version and extras hold, ignoring its marker."""

    installed = _installed_version(requirement.name)
    return installed is not None and _installed_holds(requirement, installed, seen)


def _installed_holds(requirement: Requirement, installed: str, seen: set[tuple[str, str]]) -> bool:
    """Whether *installed* meets *requirement*'s specifier and its extras hold.

    *seen* holds the ``(distribution, extra)`` pairs already being checked. A
    declared requirement that cannot be parsed or evaluated is not satisfied.
    """

    if not requirement.specifier.contains(installed, prereleases=True):
        return False
    name = canonicalize_name(requirement.name)
    for extra in sorted(requirement.extras):
        key = (name, canonicalize_name(extra))
        if key in seen:
            continue
        seen.add(key)
        for declared in metadata.requires(requirement.name) or ():
            try:
                dependency = Requirement(declared)
            except InvalidRequirement:
                return False
            marker = dependency.marker
            if marker is None:
                continue
            with_extra = marker_value(marker, {"extra": extra})
            without_extra = marker_value(marker, {"extra": ""})
            if with_extra is None or without_extra is None:
                return False
            if with_extra and not without_extra and not _holds(dependency, seen):
                return False
    return True


def _installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
