"""The match compiler's invariant self-checks are optional and gated by a flag.

The checks re-verify the compiler's own output and never change its result, so
they are disabled in normal execution.  The test suite enables them globally
(see ``tests/conftest.py``); these tests pin the gating contract and confirm the
production path — validation disabled — trusts the compiler without re-checking.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import replace

import pytest

from agm.agl.matchcompile import MatchCompiledProgram, compile_program_matches
from agm.agl.matchcompile.optional_validation import (
    match_validation_enabled,
    run_optional_validation,
    set_match_validation_enabled,
)
from agm.agl.parser import parse_program
from agm.agl.scope import resolve
from agm.agl.typecheck import check
from tests.agl.ir_harness import base_caps


@pytest.fixture
def match_validation_disabled() -> Generator[None, None, None]:
    """Run the body with the optional self-checks off, then restore the prior state."""
    previous = match_validation_enabled()
    set_match_validation_enabled(False)
    try:
        yield
    finally:
        set_match_validation_enabled(previous)


def _compiled_program(source: str) -> MatchCompiledProgram:
    result = compile_program_matches(check(resolve(parse_program(source)), base_caps()))
    assert isinstance(result.compiled, MatchCompiledProgram)
    return result.compiled


def test_optional_self_checks_are_enabled_for_the_suite() -> None:
    assert match_validation_enabled() is True


def test_run_optional_validation_runs_the_check_when_enabled() -> None:
    ran = False

    def check_() -> None:
        nonlocal ran
        ran = True

    run_optional_validation(check_)
    assert ran is True


def test_run_optional_validation_skips_the_check_when_disabled(
    match_validation_disabled: None,
) -> None:
    def check_() -> None:
        raise AssertionError("optional validation must not run when disabled")

    run_optional_validation(check_)


def test_disabled_validation_accepts_a_corrupt_artifact(
    match_validation_disabled: None,
) -> None:
    compiled = _compiled_program("case true of | true => 1 | false => 2")
    (case_id,) = tuple(compiled.cases)
    corrupt = replace(compiled.cases[case_id], reachable_action_ids=())

    # The artifact boundary does not re-check itself in production: a case whose
    # reachable-action set no longer matches its decision DAG is accepted rather
    # than rejected with ``MatchCompileInvariantError``.
    artifact = MatchCompiledProgram(compiled.checked, {case_id: corrupt})

    assert artifact.cases[case_id].reachable_action_ids == ()
