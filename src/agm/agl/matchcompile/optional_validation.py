"""Toggle for the match compiler's optional self-validation.

The match compiler carries invariant self-checks that re-verify its own output:
matrix operation-boundary consistency, occurrence-ledger integrity, decision-DAG
shape, a full semantic replay of the compile, and artifact provenance. These
checks never change the compiler's result — they only assert that a correct
compiler stayed correct — so they are disabled during normal execution.

The test harness enables them (see ``tests/conftest.py``) so that every case
compiled anywhere in the suite is validated as an invariant oracle, while
production pays none of the cost.
"""

from __future__ import annotations

from collections.abc import Callable

_ENABLED = False


def match_validation_enabled() -> bool:
    """Whether the optional match-compilation self-checks run."""
    return _ENABLED


def set_match_validation_enabled(enabled: bool) -> None:
    """Enable or disable the optional self-checks.

    Intended for the test harness only; normal execution leaves them disabled.
    """
    global _ENABLED
    _ENABLED = enabled


def run_optional_validation(check: Callable[[], None]) -> None:
    """Invoke ``check`` only when the optional self-checks are enabled."""
    if _ENABLED:
        check()
