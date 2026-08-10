"""Termination guard for monotone AgL re-export fixed points."""

from __future__ import annotations

from collections.abc import Callable


class ReexportCycleError(ValueError):
    """Raised when re-export propagation keeps growing beyond every simple module path."""


def converge_reexports(declaration_count: int, propagate: Callable[[], bool]) -> None:
    """Run a monotone re-export pass, rejecting path-expanding cycles.

    A contribution that does not revisit an export declaration crosses at most
    ``declaration_count`` declarations. One additional pass observes
    convergence. Continued growth after that can only depend on reapplying a
    declaration through a cycle.
    """
    for _ in range(declaration_count + 1):
        if not propagate():
            return
    raise ReexportCycleError("cyclic re-export expansion does not converge")
