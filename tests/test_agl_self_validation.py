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
from pathlib import Path

import pytest

from agm.agl.ir.program import ExecutableProgram
from agm.agl.ir.validate import InvalidIrError, validate_ir
from agm.agl.lower import LinkImage, lower_module, lower_program, lower_repl_entry
from agm.agl.lower.lowerer import _Lowerer
from agm.agl.matchcompile import (
    MatchCompiledModule,
    MatchCompiledProgram,
    compile_module_matches,
    compile_program_matches,
)
from agm.agl.matchcompile.normalize import MatchCompileInvariantError
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.parser import parse_program
from agm.agl.scope import resolve_module
from agm.agl.scope.program import resolve_program
from agm.agl.self_validation import self_validation_enabled
from agm.agl.typecheck import check_module
from agm.agl.typecheck.program import check_program
from tests.agl.ir_harness import base_caps, make_graph_from_files

_SOURCE = "let x = 1\nx"

# A case that specializes a constructor column, so compiling it drives every
# pattern-matrix operation that carries an optional self-check.
_MATCH_SOURCE = (
    "enum Pair\n"
    "  | pair(left: bool, right: bool)\n"
    "  | empty\n"
    "let subject: Pair = pair(left = false, right = true)\n"
    "case subject of\n"
    "  | pair(left = false, right = _) => 1\n"
    "  | pair(left = true, right = false) => 2\n"
    "  | _ => 3\n"
)


def _compiled_program(source: str) -> MatchCompiledModule:
    result = compile_module_matches(
        check_module(resolve_module(parse_program(source)), base_caps())
    )
    assert isinstance(result.compiled, MatchCompiledModule)
    return result.compiled


def _lower(source: str) -> ExecutableProgram:
    return lower_module(
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


def test_disabled_validation_accepts_a_corrupt_artifact(
    self_validation_disabled: None,
) -> None:
    compiled = _compiled_program("case true of | true => 1 | false => 2")
    (case_id,) = tuple(compiled.cases)
    corrupt = replace(compiled.cases[case_id], reachable_action_ids=())

    # The artifact boundary does not re-check itself in production: a case whose
    # reachable-action set no longer matches its decision DAG is accepted rather
    # than rejected with ``MatchCompileInvariantError``.
    artifact = MatchCompiledModule(compiled.checked, {case_id: corrupt})

    assert artifact.cases[case_id].reachable_action_ids == ()


def test_graph_lowering_trusts_the_artifact_that_already_validated_itself(
    tmp_path: Path,
) -> None:
    """A match-compiled graph is validated once, where it is built — not again per consumer."""
    graph = make_graph_from_files(tmp_path, {"entry": "case true of | true => 1 | false => 2"})
    compiled = compile_program_matches(check_program(resolve_program(graph), base_caps())).compiled
    assert isinstance(compiled, MatchCompiledProgram)
    entry_cases = compiled.cases_by_module[ENTRY_ID]
    (case_id,) = tuple(entry_cases)

    # Constructing the artifact is the checkpoint, so corruption applied behind
    # its back afterwards is rejected there ...
    corrupt = {
        module_id: dict(module_cases)
        for module_id, module_cases in compiled.cases_by_module.items()
    }
    corrupt[ENTRY_ID][case_id] = replace(entry_cases[case_id], reachable_action_ids=())
    with pytest.raises(MatchCompileInvariantError, match="reachable action ids"):
        MatchCompiledProgram(compiled.checked, corrupt)

    # ... and not re-checked by lowering, which never repeats the check.
    object.__setattr__(compiled, "cases_by_module", corrupt)
    assert lower_program(compiled).symbols


def test_program_lowering_trusts_the_artifact_that_already_validated_itself() -> None:
    """A match-compiled program is validated once, where it is built — not again per consumer."""
    source = "case true of | true => 1 | false => 2"
    compiled = _compiled_program(source)
    (case_id,) = tuple(compiled.cases)
    corrupt = dict(compiled.cases)
    corrupt[case_id] = replace(compiled.cases[case_id], reachable_action_ids=())

    # Constructing the artifact is the checkpoint, so corruption applied behind
    # its back afterwards is rejected there ...
    with pytest.raises(MatchCompileInvariantError, match="reachable action ids"):
        MatchCompiledModule(compiled.checked, corrupt)

    # ... and not re-checked by lowering, which never repeats the check.
    object.__setattr__(compiled, "cases", corrupt)
    assert lower_module(compiled, source_text=source, source_label="<test>").symbols


def test_repl_entry_lowering_trusts_the_artifact_that_already_validated_itself() -> None:
    """A REPL entry's artifact is validated at construction, not again by incremental lowering."""
    source = "case true of | true => 1 | false => 2"
    compiled = _compiled_program(source)
    (case_id,) = tuple(compiled.cases)
    corrupt = dict(compiled.cases)
    corrupt[case_id] = replace(compiled.cases[case_id], reachable_action_ids=())

    with pytest.raises(MatchCompileInvariantError, match="reachable action ids"):
        MatchCompiledModule(compiled.checked, corrupt)

    object.__setattr__(compiled, "cases", corrupt)
    entry = lower_repl_entry(
        compiled,
        image=LinkImage(),
        source_text=source,
        source_label="<repl:1>",
    )
    assert entry.program.symbols


def test_repl_entry_lowering_does_not_validate_ir_when_disabled(
    self_validation_disabled: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production REPL lowering trusts its own output: structural IR validation is test-only."""
    _break_symbol_table(monkeypatch)

    entry = lower_repl_entry(
        _compiled_program(_SOURCE),
        image=LinkImage(),
        source_text=_SOURCE,
        source_label="<repl:1>",
    )

    # Returned rather than rejected, even though the IR is structurally invalid.
    assert entry.program.symbols == {}
    with pytest.raises(InvalidIrError):
        validate_ir(entry.program)


def test_repl_entry_lowering_validates_ir_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the flag on (the suite default), every REPL lowering is an IR invariant oracle."""
    _break_symbol_table(monkeypatch)

    with pytest.raises(InvalidIrError):
        lower_repl_entry(
            _compiled_program(_SOURCE),
            image=LinkImage(),
            source_text=_SOURCE,
            source_label="<repl:1>",
        )


def test_production_match_compilation_takes_the_unvalidated_path(
    self_validation_disabled: None,
    tmp_path: Path,
) -> None:
    """Compiling and lowering with the flag off skips every optional check.

    This drives each guarded site — the matrix operations, both artifact
    boundaries, the rejected stage result, and whole-program lowering — down its
    production branch, where the compiler's own output is trusted as produced.
    """
    program_artifact = _compiled_program(_MATCH_SOURCE)
    assert program_artifact.cases

    rejected = compile_module_matches(
        check_module(resolve_module(parse_program("case true of | true => 1")), base_caps())
    )
    assert rejected.compiled is None
    assert rejected.issues != ()

    graph = make_graph_from_files(tmp_path, {"entry": _MATCH_SOURCE})
    program_result = compile_program_matches(check_program(resolve_program(graph), base_caps()))
    assert isinstance(program_result.compiled, MatchCompiledProgram)

    lowered = lower_program(program_result.compiled)
    assert lowered.symbols


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
