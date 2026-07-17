"""AgL's invariant self-checks are optional and gated by a single flag.

The checks re-verify artifacts the compiler itself just produced — match
compilation and the structural validation of the lowered IR — and never change a
result, so they are disabled in normal execution.  The test suite enables them
globally (see ``tests/conftest.py``); these tests pin the gating contract and
confirm the production path — validation disabled — trusts the compiler and the
lowerer without re-checking either.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from agm.agl.ir.program import ExecutableProgram
from agm.agl.ir.validate import InvalidIrError, validate_ir
from agm.agl.lower import lower_program
from agm.agl.lower.lowerer import _Lowerer
from agm.agl.matchcompile import MatchCompiledProgram, compile_program_matches
from agm.agl.parser import parse_program
from agm.agl.scope import resolve
from agm.agl.self_validation import run_optional_validation, self_validation_enabled
from agm.agl.typecheck import check
from tests.agl.ir_harness import base_caps

_SOURCE = "let x = 1\nx"


def _compiled_program(source: str) -> MatchCompiledProgram:
    result = compile_program_matches(check(resolve(parse_program(source)), base_caps()))
    assert isinstance(result.compiled, MatchCompiledProgram)
    return result.compiled


def _lower(source: str) -> ExecutableProgram:
    return lower_program(
        _compiled_program(source),
        source_text=source,
        source_label="<test>",
    )


def _break_symbol_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every lowering emit IR whose loads dangle off an empty symbol table."""
    real_lower = _Lowerer.lower

    def corrupt_lower(self: _Lowerer) -> ExecutableProgram:
        return replace(real_lower(self), symbols={})

    monkeypatch.setattr(_Lowerer, "lower", corrupt_lower)


def test_optional_self_checks_are_enabled_for_the_suite() -> None:
    assert self_validation_enabled() is True


def test_run_optional_validation_runs_the_check_when_enabled() -> None:
    ran = False

    def check_() -> None:
        nonlocal ran
        ran = True

    run_optional_validation(check_)
    assert ran is True


def test_run_optional_validation_skips_the_check_when_disabled(
    self_validation_disabled: None,
) -> None:
    def check_() -> None:
        raise AssertionError("optional validation must not run when disabled")

    run_optional_validation(check_)


def test_disabled_validation_accepts_a_corrupt_artifact(
    self_validation_disabled: None,
) -> None:
    compiled = _compiled_program("case true of | true => 1 | false => 2")
    (case_id,) = tuple(compiled.cases)
    corrupt = replace(compiled.cases[case_id], reachable_action_ids=())

    # The artifact boundary does not re-check itself in production: a case whose
    # reachable-action set no longer matches its decision DAG is accepted rather
    # than rejected with ``MatchCompileInvariantError``.
    artifact = MatchCompiledProgram(compiled.checked, {case_id: corrupt})

    assert artifact.cases[case_id].reachable_action_ids == ()


def test_lowering_does_not_validate_ir_when_disabled(
    self_validation_disabled: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production lowering trusts its own output: structural IR validation is test-only."""
    _break_symbol_table(monkeypatch)

    program = _lower(_SOURCE)

    # Returned rather than rejected, even though the IR is structurally invalid.
    assert program.symbols == {}
    with pytest.raises(InvalidIrError):
        validate_ir(program)


def test_lowering_validates_ir_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the flag on (the suite default), every lowering is an IR invariant oracle."""
    _break_symbol_table(monkeypatch)

    with pytest.raises(InvalidIrError):
        _lower(_SOURCE)
