"""AgL's invariant self-checks are optional and gated by a single flag.

The checks re-verify artifacts the compiler itself just produced — the
typechecker's own closed-output boundary, match compilation, and the structural
validation of the lowered IR — and never change a result, so they are disabled
in normal execution.  The test suite enables them globally (see
``tests/conftest.py``); these tests pin the gating contract and confirm the
production path — validation disabled — trusts the checker, the compiler, and
the lowerer without re-checking any of them.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agm.agl.ir.validate import InvalidIrError, validate_ir
from agm.agl.lower import LinkImage, lower_program, lower_repl_program
from agm.agl.lower.lowerer import _LinkState
from agm.agl.matchcompile import MatchCompiledProgram, compile_program_matches
from agm.agl.matchcompile.normalize import MatchCompileInvariantError
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.repl import ReplSession
from agm.agl.scope.program import resolve_program
from agm.agl.self_validation import self_validation_enabled
from agm.agl.semantics.type_table import TypeDef, TypeTable
from agm.agl.semantics.types import InferenceVarType
from agm.agl.typecheck.env import assert_checked_module_closed
from agm.agl.typecheck.program import check_program
from tests.agl.ir_harness import (
    _checked_program,
    _compiled_program,
    base_caps,
    make_graph_from_files,
)
from tests.agl.match_reference import case_sites
from tests.agl.module_graph import resolve_and_check_entry

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


class _SwallowingSymbols(dict):
    """A symbol table that silently discards every write.

    ``lower_program`` builds its final ``ExecutableProgram.symbols`` straight
    from the shared ``_LinkState.symbols`` it links into, so swapping that
    dict for one that never actually stores anything reproduces "every
    lowering emits IR whose loads dangle off an empty symbol table" without
    reaching into ``_Lowerer`` internals.
    """

    def __setitem__(self, key: object, value: object) -> None:
        pass


def _broken_link() -> _LinkState:
    link = _LinkState()
    link.symbols = _SwallowingSymbols()
    return link


def test_optional_self_checks_are_enabled_for_the_suite() -> None:
    assert self_validation_enabled() is True


def test_disabled_validation_accepts_a_corrupt_artifact(
    self_validation_disabled: None,
) -> None:
    compiled = _compiled_program("case true of | true => 1 | false => 2")
    entry_sites = compiled.sites_by_module[ENTRY_ID]
    (case_id,) = tuple(case_sites(entry_sites))
    corrupt = replace(case_sites(entry_sites)[case_id], reachable_action_ids=())

    # The artifact boundary does not re-check itself in production: a case whose
    # reachable-action set no longer matches its decision DAG is accepted rather
    # than rejected with ``MatchCompileInvariantError``.
    artifact = MatchCompiledProgram(
        compiled.checked, {**compiled.sites_by_module, ENTRY_ID: {case_id: corrupt}}
    )

    assert case_sites(artifact.sites_by_module[ENTRY_ID])[case_id].reachable_action_ids == ()


def test_graph_lowering_trusts_the_artifact_that_already_validated_itself(
    tmp_path: Path,
) -> None:
    """A match-compiled graph is validated once, where it is built — not again per consumer."""
    graph = make_graph_from_files(tmp_path, {"entry": "case true of | true => 1 | false => 2"})
    compiled = compile_program_matches(check_program(resolve_program(graph), base_caps())).compiled
    assert isinstance(compiled, MatchCompiledProgram)
    entry_cases = case_sites(compiled.sites_by_module[ENTRY_ID])
    (case_id,) = tuple(entry_cases)

    # Constructing the artifact is the checkpoint, so corruption applied behind
    # its back afterwards is rejected there ...
    corrupt = {
        module_id: dict(module_sites)
        for module_id, module_sites in compiled.sites_by_module.items()
    }
    corrupt[ENTRY_ID][case_id] = replace(entry_cases[case_id], reachable_action_ids=())
    with pytest.raises(MatchCompileInvariantError, match="reachable action ids"):
        MatchCompiledProgram(compiled.checked, corrupt)

    # ... and not re-checked by lowering, which never repeats the check.
    object.__setattr__(compiled, "sites_by_module", corrupt)
    assert lower_program(compiled).symbols


def test_repl_entry_lowering_trusts_the_artifact_that_already_validated_itself() -> None:
    """A REPL entry's artifact is validated at construction, not again by incremental lowering."""
    source = "case true of | true => 1 | false => 2"
    compiled = _compiled_program(source)
    entry_sites = compiled.sites_by_module[ENTRY_ID]
    (case_id,) = tuple(case_sites(entry_sites))
    corrupt = dict(entry_sites)
    corrupt[case_id] = replace(case_sites(entry_sites)[case_id], reachable_action_ids=())

    with pytest.raises(MatchCompileInvariantError, match="reachable action ids"):
        MatchCompiledProgram(compiled.checked, {**compiled.sites_by_module, ENTRY_ID: corrupt})

    object.__setattr__(compiled, "sites_by_module", {**compiled.sites_by_module, ENTRY_ID: corrupt})
    entry = lower_repl_program(compiled, image=LinkImage(), source_text=source)
    assert entry.program.symbols


def _broken_image() -> LinkImage:
    image = LinkImage()
    image._state = _broken_link()
    return image


def test_repl_entry_lowering_does_not_validate_ir_when_disabled(
    self_validation_disabled: None,
) -> None:
    """Production REPL lowering trusts its own output: structural IR validation is test-only."""
    entry = lower_repl_program(
        _compiled_program(_SOURCE), image=_broken_image(), source_text=_SOURCE
    )

    # Returned rather than rejected, even though the IR is structurally invalid.
    assert entry.program.symbols == {}
    with pytest.raises(InvalidIrError):
        validate_ir(entry.program)


def test_repl_entry_lowering_validates_ir_when_enabled() -> None:
    """With the flag on (the suite default), every REPL lowering is an IR invariant oracle."""
    with pytest.raises(InvalidIrError):
        lower_repl_program(_compiled_program(_SOURCE), image=_broken_image(), source_text=_SOURCE)


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
    assert case_sites(program_artifact.sites_by_module[ENTRY_ID])

    rejected = compile_program_matches(_checked_program("case true of | true => 1"))
    assert rejected.compiled is None
    assert rejected.issues != ()

    graph = make_graph_from_files(tmp_path, {"entry": _MATCH_SOURCE})
    program_result = compile_program_matches(check_program(resolve_program(graph), base_caps()))
    assert isinstance(program_result.compiled, MatchCompiledProgram)

    lowered = lower_program(program_result.compiled)
    assert lowered.symbols


def test_lowering_does_not_validate_ir_when_disabled(
    self_validation_disabled: None,
) -> None:
    """Production lowering trusts its own output: structural IR validation is test-only."""
    program = lower_program(_compiled_program(_SOURCE), _link=_broken_link())

    # Returned rather than rejected, even though the IR is structurally invalid.
    assert program.symbols == {}
    with pytest.raises(InvalidIrError):
        validate_ir(program)


def test_lowering_validates_ir_when_enabled() -> None:
    """With the flag on (the suite default), every lowering is an IR invariant oracle."""
    with pytest.raises(InvalidIrError):
        lower_program(_compiled_program(_SOURCE), _link=_broken_link())


# ---------------------------------------------------------------------------
# Typechecker self-checks: the closed-output boundary and its call sites
# ---------------------------------------------------------------------------


def test_disabled_validation_still_seals_the_checked_type_env(
    self_validation_disabled: None,
) -> None:
    """Sealing is functional state, not a self-check, so it survives being disabled.

    ``TypeEnvironment.seal`` freezes the environment (gating further mutation and
    enabling memoization/seeding) independently of whether it also re-verifies
    itself; a checked module's ``type_env`` must come out sealed either way.
    """
    checked = resolve_and_check_entry(_SOURCE, base_caps())

    assert checked.type_env.is_sealed is True


def test_disabled_validation_skips_the_inference_region_leak_check(
    self_validation_disabled: None,
) -> None:
    """A region that actually allocates inference variables is trusted once resolved.

    Calling a generic function forces the checker to allocate a flexible
    inference variable for its type parameter and resolve it from the argument;
    the finalized-region leak check that re-walks the result is a self-check
    only, so disabling it does not stop the call from checking normally.
    """
    source = "def id[T](value: T) -> T = value\nlet result = id(1)\nresult"
    checked = resolve_and_check_entry(source, base_caps())

    assert checked.node_types


def test_disabled_validation_skips_the_extern_target_leak_check(
    self_validation_disabled: None,
    tmp_path: Path,
) -> None:
    """A generic extern referenced first-class also carries a resolved leak check.

    Taking a generic extern function as a value (rather than calling it
    directly) routes its result type through ``_zonk_extern_targets``, which
    re-verifies the zonked provenance is free of inference variables. That
    re-verification is a self-check only: disabling it does not stop the
    reference from resolving to a concrete call site.
    """
    source = "extern def id[T](value: T) -> T\nlet apply: (int) -> int = id\napply(1)"
    origin_path = tmp_path / "self_validation.agl"
    origin_path.with_suffix(".py").write_text("")
    checked = resolve_and_check_entry(source, base_caps(), origin_path=origin_path)

    assert [site.callee for site in checked.call_sites] == ["id"]


def test_disabled_validation_skips_the_repl_entry_closure_check(
    self_validation_disabled: None,
) -> None:
    """Promoting a REPL entry trusts its own checked output without re-verifying it."""
    result = ReplSession().eval_entry("let x = 1")

    assert result.ok


def test_disabled_validation_accepts_a_re_registered_typedef_that_actually_conflicts(
    self_validation_disabled: None,
) -> None:
    """Re-registering under an existing key is always accepted once disabled.

    In production, re-registering the identical declaration is expected (e.g.
    the program pre-pass and the per-module check both build the same module);
    the full recursive-dataclass comparison that would additionally reject a
    genuinely *different* redefinition is a self-check only, so a conflicting
    re-registration is silently accepted too, rather than raising.
    """
    table = TypeTable()
    table.register(TypeDef(kind="record", name="Box", module_id=ENTRY_ID))

    table.register(TypeDef(kind="record", name="Box", module_id=ENTRY_ID, type_params=("T",)))

    assert table.get(ENTRY_ID, "Box") == TypeDef(kind="record", name="Box", module_id=ENTRY_ID)


def test_enabled_validation_rejects_a_conflicting_typedef_re_registration() -> None:
    """With the flag on (the suite default), a genuinely conflicting redefinition is caught."""
    table = TypeTable()
    table.register(TypeDef(kind="record", name="Box", module_id=ENTRY_ID))

    with pytest.raises(AssertionError, match="conflicting TypeDef registration"):
        table.register(TypeDef(kind="record", name="Box", module_id=ENTRY_ID, type_params=("T",)))


def test_closed_output_covers_explicit_builtin_targets() -> None:
    """A leaked inference variable in ``explicit_builtin_targets`` is caught like any sibling.

    ``explicit_builtin_targets`` is the side table for an explicit ``::[T]``
    type argument on a builtin call (``print``, ``ask``, ``exec``, ...); it must
    be walked by the closed-output boundary exactly like every other
    type-bearing side table.
    """
    checked = resolve_and_check_entry(_SOURCE, base_caps())
    leaked = replace(checked, explicit_builtin_targets={-1: InferenceVarType("leak")})

    with pytest.raises(AssertionError, match="inference variable leaked"):
        assert_checked_module_closed(leaked)
