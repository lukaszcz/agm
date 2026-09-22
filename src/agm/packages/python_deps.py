"""Package Python requirements: syncing, plus checks for ``pkg check`` and companion loading.

The activation index is the source of truth: the running interpreter's
environment is synced to the union of every active package's
``[python] dependencies``. Nothing is ever uninstalled.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from agm.core import pyenv
from agm.packages.model import PackageInfo


def python_dependencies(packages: Iterable[PackageInfo]) -> tuple[str, ...]:
    """Union of *packages*' requirements in package then manifest order; duplicates once."""

    specs: dict[str, None] = {}
    for package in packages:
        for spec in package.manifest.python_dependencies:
            specs[spec] = None
    return tuple(specs)


def unsatisfied_python_dependencies(
    specs: Iterable[str],
) -> tuple[tuple[str, pyenv.RequirementStatus], ...]:
    """Each of *specs* the running interpreter's environment does not satisfy, with its status."""

    statuses = ((spec, pyenv.requirement_status(spec)) for spec in specs)
    return tuple((spec, status) for spec, status in statuses if not status.satisfied)


def sync_python_dependencies(specs: Sequence[str]) -> tuple[str, ...]:
    """Install all *specs* jointly when any is unsatisfied; return the unsatisfied ones.

    The whole union goes to the installer so it resolves every requirement
    together, failing on conflicts rather than downgrading a satisfied one.
    Honours dry-run; raises :class:`~agm.core.pyenv.RequirementInstallError`.
    """

    unsatisfied = tuple(spec for spec, _ in unsatisfied_python_dependencies(specs))
    if unsatisfied:
        pyenv.install_requirements(specs)
    return unsatisfied
