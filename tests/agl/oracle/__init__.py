"""Differential oracle harness for M2/M3 — compares legacy AST interpreter vs IR evaluator.

Public API
----------
- :func:`assert_oracle_agrees` — run both evaluators and assert they agree.
- :func:`assert_oracle_raises` — assert both evaluators raise equivalent exceptions.
"""

from tests.agl.oracle.harness import assert_oracle_agrees, assert_oracle_raises

__all__ = ["assert_oracle_agrees", "assert_oracle_raises"]
