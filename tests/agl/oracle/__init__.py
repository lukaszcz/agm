"""Differential oracle harness for M2 — compares legacy AST interpreter vs IR evaluator.

Public API
----------
- :func:`assert_oracle_agrees` — run both evaluators and assert they agree.
"""

from tests.agl.oracle.harness import assert_oracle_agrees

__all__ = ["assert_oracle_agrees"]
