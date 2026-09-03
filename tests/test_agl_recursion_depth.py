"""Deep recursion raises a *catchable* AgL ``RecursionError``, never a host crash.

The tree-walker consumes several Python stack frames per AgL call, so without
intervention a deep recursion (especially the non-tail shape, where the
recursive call is an operand and its frame stays live) hits Python's own
recursion limit and escapes as an uncaught Python ``RecursionError`` before the
AgL ``max_call_depth`` guard ever fires. ``IrInterpreter.run`` raises Python's
limit so the AgL guard governs, and converts any Python ``RecursionError`` that
still escapes (its limit is capped) into the same catchable AgL exception.

The static passes need the same contract for a different reason: they recurse
over the syntax tree, so source nested deeply enough exhausts the stack before
any pass can report anything.  ``frontend_recursion_boundary`` converts that
into an ordinary pre-execution diagnostic for every pass the pipeline and the
REPL drive.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from agm.agl.diagnostics import AglError, Diagnostic
from agm.agl.eval import ir_interpreter
from agm.agl.pipeline import PipelineDriver, RunResult
from agm.agl.recursion import NestingTooDeepError, frontend_recursion_boundary
from agm.agl.repl.session import ReplSession
from agm.agl.semantics.values import IntValue
from tests._agl_helpers import run_inline_command

# A non-tail recursive helper plus a variant guarded by a ``try``/``catch``.
_PRELUDE = """
def sum_to(n: int) -> int =
  if n == 0 => 0 else => n + sum_to(n - 1)

def guarded(d: int) -> int =
  try sum_to(d)
  catch RecursionError as e => e.limit
"""


def _run(initializer: str, *, depth: int, max_call_depth: int):
    # ``out`` is bound then used so the block ends in an expression, keeping
    # ``out`` a public binding the caller can inspect.
    source = f"{_PRELUDE}let depth: int = {depth}\nlet out: int = {initializer}\nprint(out)\n"
    driver = PipelineDriver(default_call_depth_limit=max_call_depth)
    return run_inline_command(driver, source)


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


def _nested_parens(depth: int) -> str:
    return f"let x: int = {'(' * depth}1{')' * depth}\nprint(x)\n"


def _nested_type_args(depth: int) -> str:
    return f"let x: {'array[' * depth}int{']' * depth} = []\nprint(1)\n"


def _nested_calls(depth: int) -> str:
    return f"def f(n: int) -> int = n\nlet x: int = {'f(' * depth}1{')' * depth}\nprint(x)\n"


def _nested_if(depth: int) -> str:
    body = "0"
    for _ in range(depth):
        body = f"(if 1 == 1 => 1 else => {body})"
    return f"let x: int = {body}\nprint(x)\n"


def _nested_case(depth: int) -> str:
    body = "0"
    for _ in range(depth):
        body = f"(case 1 of | 0 => 0 | _ => {body})"
    return f"let x: int = {body}\nprint(x)\n"


def _operator_chain(terms: int) -> str:
    return f"let x: int = {'+'.join(['1'] * terms)}\nprint(x)\n"


def _check_source(source: str) -> RunResult:
    """Run every static frontend pass over *source* without executing it."""
    return run_inline_command(PipelineDriver(), source, check_only=True)


def _raise_recursion_error(*_args: object, **_kwargs: object) -> object:
    raise RecursionError("maximum recursion depth exceeded")


class TestFrontendNestingDegradesToDiagnostic:
    """Source too deeply nested for the frontend is a diagnostic, never a host crash.

    Which pass runs out of stack first depends on the shape — and, for any one
    shape, on how many frames the host already has on the stack — so these
    depths are chosen well past the shallowest cliff rather than at it.  The
    contract under test is shape- and pass-independent: whichever pass gives
    out, the user gets an ordinary pre-execution diagnostic.
    """

    @pytest.mark.parametrize(
        ("build", "depth"),
        [
            (_nested_parens, 600),
            (_nested_type_args, 600),
            (_nested_calls, 600),
            (_nested_if, 400),
            (_nested_case, 400),
            (_operator_chain, 1200),
        ],
        ids=["parens", "type-args", "calls", "if", "case", "operator-chain"],
    )
    def test_deep_nesting_reports_a_diagnostic(
        self, build: Callable[[int], str], depth: int
    ) -> None:
        result = _check_source(build(depth))
        assert not result.ok
        assert result.error is None
        assert result.diagnostics
        assert all(isinstance(d, Diagnostic) for d in result.diagnostics)


class TestEveryGuardedPassReportsStackExhaustion:
    """Each static pass the pipeline drives converts stack exhaustion on its own.

    A depth-based test only reaches whichever pass happens to give out first,
    which leaves the later passes' handling untested — that is exactly how
    lowering came to have no handler at all.  Injecting the failure at each pass
    boundary pins every one of them independently of the host's stack.
    """

    @pytest.mark.parametrize(
        ("module", "name"),
        [
            ("agm.agl.modules.loader", "parse_entry_module"),
            ("agm.agl.modules.loader", "build_repl_graph"),
            ("agm.agl.scope.program", "resolve_program"),
            ("agm.agl.typecheck.program", "check_program"),
            ("agm.agl.lower", "lower_program"),
        ],
        ids=["parse", "module-load", "scope", "typecheck", "lower"],
    )
    def test_pass_stack_exhaustion_becomes_a_diagnostic(
        self, monkeypatch: pytest.MonkeyPatch, module: str, name: str
    ) -> None:
        monkeypatch.setattr(f"{module}.{name}", _raise_recursion_error)
        result = _check_source("print(1)\n")
        assert not result.ok
        assert result.error is None
        assert result.diagnostics
        assert all(isinstance(d, Diagnostic) for d in result.diagnostics)

    def test_repl_lowering_stack_exhaustion_becomes_a_diagnostic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = ReplSession()
        monkeypatch.setattr("agm.agl.lower.lower_repl_program", _raise_recursion_error)
        assert session.eval_entry("let deep: int = 1 + 1").diagnostics
        # The failed entry rolled back cleanly: the session still evaluates.
        monkeypatch.undo()
        assert not session.eval_entry("1 + 1").diagnostics


class TestFrontendRecursionBoundary:
    """The frontend boundary converts Python's ``RecursionError`` into an AgL error."""

    def test_python_recursion_error_becomes_an_agl_error(self) -> None:
        def blow(n: int) -> int:
            return blow(n + 1)

        with pytest.raises(NestingTooDeepError) as excinfo:
            with frontend_recursion_boundary():
                blow(0)
        assert isinstance(excinfo.value, AglError)
        assert isinstance(excinfo.value.to_diagnostic(), Diagnostic)

    def test_other_errors_pass_through_unchanged(self) -> None:
        with pytest.raises(ValueError):
            with frontend_recursion_boundary():
                raise ValueError("unrelated")


class TestReplNestingDegradesToDiagnostic:
    """The REPL reports over-deep source the same way, without killing the session."""

    def test_deep_operator_chain_reports_a_diagnostic(self) -> None:
        session = ReplSession()
        result = session.eval_entry(f"let deep: int = {'+'.join(['1'] * 1200)}")
        assert result.diagnostics
        assert all(isinstance(d, Diagnostic) for d in result.diagnostics)
        # The session survives: an ordinary entry still evaluates afterwards.
        assert not session.eval_entry("1 + 1").diagnostics
