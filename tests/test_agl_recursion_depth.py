"""Deep recursion raises a *catchable* AgL ``RecursionError``, never a host crash.

The tree-walker consumes several Python stack frames per AgL call, so without
intervention a deep recursion (especially the non-tail shape, where the
recursive call is an operand and its frame stays live) hits Python's own
recursion limit and escapes as an uncaught Python ``RecursionError`` before the
AgL ``max_call_depth`` guard ever fires. ``IrInterpreter.run`` raises Python's
limit so the AgL guard governs, and converts any Python ``RecursionError`` that
still escapes (its limit is capped) into the same catchable AgL exception.
"""

from __future__ import annotations

import pytest

from agm.agl.eval import ir_interpreter
from agm.agl.pipeline import PipelineDriver
from agm.agl.semantics.values import IntValue

# A non-tail recursive helper plus a variant guarded by a ``try``/``catch``.
_PRELUDE = """
param depth: int

def sum_to(n: int) -> int =
  if n == 0 => 0 else => n + sum_to(n - 1)

def guarded(d: int) -> int =
  try sum_to(d)
  catch RecursionError as e => e.limit
"""


def _run(initializer: str, *, depth: int, max_call_depth: int):
    # ``out`` is bound then used so the block ends in an expression, keeping
    # ``out`` a public binding the caller can inspect.
    source = f"{_PRELUDE}let out: int = {initializer}\nprint(out)\n"
    driver = PipelineDriver(default_call_depth_limit=max_call_depth)
    return driver.run(source, param_values={"depth": depth})


class TestGuardIsAuthoritative:
    """With the recursion limit raised, the AgL guard trips first and is catchable."""

    def test_non_tail_recursion_error_is_catchable(self) -> None:
        result = _run("guarded(depth)", depth=100_000, max_call_depth=300)
        assert result.ok, [d.message for d in result.diagnostics]
        assert result.bindings["out"] == IntValue(300)

    def test_uncaught_non_tail_recursion_surfaces_the_agl_exception(self) -> None:
        result = _run("sum_to(depth)", depth=100_000, max_call_depth=300)
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "RecursionError"
        assert result.error.fields["limit"] == 300


class TestPythonRecursionErrorBackstop:
    """When Python's (capped) limit is hit before the guard, it is still catchable.

    The cap is lowered so ``run`` cannot raise Python's limit high enough for the
    guard to be reached, forcing the deep recursion to hit Python's limit first —
    the pathological case the backstop exists for — at a shallow, fast depth.
    """

    @pytest.fixture(autouse=True)
    def _cap_recursion_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ir_interpreter, "_MAX_PYTHON_RECURSION_LIMIT", 1500)

    def test_backstop_recursion_error_is_catchable(self) -> None:
        result = _run("guarded(depth)", depth=1_000_000, max_call_depth=1_000_000)
        assert result.ok, [d.message for d in result.diagnostics]
        assert result.bindings["out"] == IntValue(1_000_000)

    def test_uncaught_backstop_recursion_error_surfaces_the_agl_exception(self) -> None:
        result = _run("sum_to(depth)", depth=1_000_000, max_call_depth=1_000_000)
        assert not result.ok
        assert result.error is not None
        assert result.error.type_name == "RecursionError"
