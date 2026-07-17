"""Toggle for AgL's optional self-validation.

AgL carries invariant self-checks that re-verify artifacts the compiler itself
just produced: the match compiler's matrix operation-boundary consistency,
occurrence-ledger integrity, decision-DAG shape, semantic replay and artifact
provenance, plus the structural validation of the lowered execution IR. None of
these checks change a result — they only assert that a correct compiler stayed
correct — so they are disabled during normal execution.

Every call site guards its check with ``if self_validation_enabled():``, so that
a disabled check costs one global read: neither the check nor any state recorded
solely for it is built on the production path.

An artifact validates itself once, where it is constructed; its consumers trust
it rather than re-checking it, so each invariant is verified exactly once per
artifact.

The test harness enables them (see ``tests/conftest.py``) so that every case
compiled and every program lowered anywhere in the suite doubles as an invariant
oracle.
"""

from __future__ import annotations

_ENABLED = False


def self_validation_enabled() -> bool:
    """Whether the optional AgL self-checks run."""
    return _ENABLED


def set_self_validation_enabled(enabled: bool) -> None:
    """Enable or disable the optional self-checks.

    Intended for the test harness only; normal execution leaves them disabled.
    """
    global _ENABLED
    _ENABLED = enabled
