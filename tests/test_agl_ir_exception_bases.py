"""Runtime ancestry: ``NominalDescriptor.base`` links and ``nominal_conforms``."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from agm.agl.ir.ids import NominalId, SourceId
from agm.agl.ir.program import (
    ExecutableModule,
    ExecutableProgram,
    NominalDescriptor,
    NominalKind,
    SourceFile,
    nominal_conforms,
)
from agm.agl.ir.reserved_nominals import require_reserved_nominal_id
from agm.agl.ir.validate import InvalidIrError, validate_ir
from agm.agl.modules.ids import ENTRY_ID
from tests.agl.ir_harness import lower_inline_ir

# ---------------------------------------------------------------------------
# Descriptor bases from lowering
# ---------------------------------------------------------------------------


def _descriptor(name: str, nominals: Mapping[NominalId, NominalDescriptor]) -> NominalDescriptor:
    matches = [d for d in nominals.values() if d.declared_name == name and d.bears_name_path]
    assert len(matches) == 1, f"expected exactly one live descriptor named {name!r}: {matches}"
    return matches[0]


def test_user_exception_chain_records_base_links() -> None:
    """A user `extends` chain links each descriptor to its declared parent."""
    source = """\
exception Problem extends Exception
  code: int
exception Detailed extends Problem
  detail: text
()
"""
    program = lower_inline_ir(source)
    problem = _descriptor("Problem", program.nominals)
    detailed = _descriptor("Detailed", program.nominals)
    root = _descriptor("Exception", program.nominals)

    assert detailed.base == problem.nominal
    assert problem.base == root.nominal
    assert root.base is None


def test_builtin_exception_extends_root() -> None:
    """A standard-library builtin exception's base is the loaded root's identity."""
    source = """\
let e = IndexError(message = "m", index = 1, length = 0)
()
"""
    program = lower_inline_ir(source)
    index_error = _descriptor("IndexError", program.nominals)
    root = _descriptor("Exception", program.nominals)

    assert index_error.base == root.nominal
    assert root.base is None


def test_base_closure_over_superseded_reserved_fallback() -> None:
    """A kept exception's base, skipped as a superseded reserved fallback, still gets a descriptor.

    Importing only ``std/errors`` (not ``std/agent``) supersedes the reserved
    "Exception" root -- the source declaration bears that name now -- while
    the reserved "AgentCallError" descriptor is still emitted (``std/agent``
    is not loaded, so it is not superseded) with its base pointing at the
    now-skipped reserved "Exception" identity. The closure step must still
    add that identity's descriptor, non-name-bearing, for runtime ancestry.
    """
    source = """\
import std/errors
()
"""
    program = lower_inline_ir(source, default_stdlib=False)

    reserved_exception_id = NominalId(require_reserved_nominal_id("Exception"))
    reserved_agent_call_error_id = NominalId(require_reserved_nominal_id("AgentCallError"))

    agent_call_error = program.nominals[reserved_agent_call_error_id]
    assert agent_call_error.base == reserved_exception_id

    closed_root = program.nominals[reserved_exception_id]
    assert closed_root.kind is NominalKind.EXCEPTION
    assert closed_root.base is None
    assert closed_root.bears_name_path is False

    # The source declaration is the one that now bears the "Exception" name.
    live_root = _descriptor("Exception", program.nominals)
    assert live_root.nominal != reserved_exception_id
    assert live_root.base is None


# ---------------------------------------------------------------------------
# nominal_conforms
# ---------------------------------------------------------------------------


def _exc(nominal_id: int, *, base: NominalId | None = None, name: str = "E") -> NominalDescriptor:
    return NominalDescriptor(
        NominalId(nominal_id), ENTRY_ID, (), name, NominalKind.EXCEPTION, base=base
    )


def test_nominal_conforms_exact_match() -> None:
    root = _exc(1, name="Exception")
    nominals = {root.nominal: root}
    assert nominal_conforms(nominals, root.nominal, {root.nominal})


def test_nominal_conforms_one_level_ancestor() -> None:
    root = _exc(1, name="Exception")
    child = _exc(2, base=root.nominal, name="Problem")
    nominals = {root.nominal: root, child.nominal: child}
    assert nominal_conforms(nominals, child.nominal, {root.nominal})
    assert not nominal_conforms(nominals, root.nominal, {child.nominal})


def test_nominal_conforms_multi_level_ancestor() -> None:
    root = _exc(1, name="Exception")
    mid = _exc(2, base=root.nominal, name="Problem")
    leaf = _exc(3, base=mid.nominal, name="Detailed")
    nominals = {root.nominal: root, mid.nominal: mid, leaf.nominal: leaf}
    assert nominal_conforms(nominals, leaf.nominal, {root.nominal})
    assert nominal_conforms(nominals, leaf.nominal, {mid.nominal})
    assert nominal_conforms(nominals, leaf.nominal, {leaf.nominal})


def test_nominal_conforms_unrelated_types() -> None:
    root = _exc(1, name="Exception")
    left = _exc(2, base=root.nominal, name="Left")
    right = _exc(3, base=root.nominal, name="Right")
    nominals = {root.nominal: root, left.nominal: left, right.nominal: right}
    assert not nominal_conforms(nominals, left.nominal, {right.nominal})


def test_nominal_conforms_root_only_matches_itself() -> None:
    root = _exc(1, name="Exception")
    other = _exc(2, name="Other")
    nominals = {root.nominal: root, other.nominal: other}
    assert nominal_conforms(nominals, root.nominal, {root.nominal})
    assert not nominal_conforms(nominals, root.nominal, {other.nominal})


# ---------------------------------------------------------------------------
# Deep validation of bases
# ---------------------------------------------------------------------------


def _program_with_nominals(nominals: dict[NominalId, NominalDescriptor]) -> ExecutableProgram:
    sid = SourceId(0)
    return ExecutableProgram(
        entry_module=ENTRY_ID,
        modules={ENTRY_ID: ExecutableModule(module_id=ENTRY_ID, initializers=())},
        symbols={},
        nominals=nominals,
        sources={sid: SourceFile(display_name="<test>", normalized_text=" ")},
    )


def test_validate_accepts_acyclic_exception_base_chain() -> None:
    root = _exc(1, name="Exception")
    mid = _exc(2, base=root.nominal, name="Problem")
    leaf = _exc(3, base=mid.nominal, name="Detailed")
    program = _program_with_nominals({d.nominal: d for d in (root, mid, leaf)})
    validate_ir(program, deep=True)  # no exception


def test_validate_rejects_exception_base_referencing_missing_nominal() -> None:
    leaf = _exc(1, base=NominalId(99), name="Detailed")
    program = _program_with_nominals({leaf.nominal: leaf})
    with pytest.raises(InvalidIrError):
        validate_ir(program, deep=True)


def test_validate_rejects_exception_base_that_is_not_an_exception() -> None:
    record = NominalDescriptor(NominalId(1), ENTRY_ID, (), "Point", NominalKind.RECORD)
    leaf = _exc(2, base=record.nominal, name="Detailed")
    program = _program_with_nominals({record.nominal: record, leaf.nominal: leaf})
    with pytest.raises(InvalidIrError):
        validate_ir(program, deep=True)


def test_validate_rejects_cyclic_exception_base_chain() -> None:
    a = _exc(1, base=NominalId(2), name="A")
    b = _exc(2, base=NominalId(1), name="B")
    program = _program_with_nominals({a.nominal: a, b.nominal: b})
    with pytest.raises(InvalidIrError):
        validate_ir(program, deep=True)


def test_validate_rejects_record_descriptor_with_base() -> None:
    root = _exc(1, name="Exception")
    record = NominalDescriptor(
        NominalId(2), ENTRY_ID, (), "Point", NominalKind.RECORD, base=root.nominal
    )
    program = _program_with_nominals({root.nominal: root, record.nominal: record})
    with pytest.raises(InvalidIrError):
        validate_ir(program, deep=True)


def test_validate_rejects_enum_descriptor_with_base() -> None:
    root = _exc(1, name="Exception")
    enum = NominalDescriptor(
        NominalId(2), ENTRY_ID, (), "Color", NominalKind.ENUM, base=root.nominal, variants=()
    )
    program = _program_with_nominals({root.nominal: root, enum.nominal: enum})
    with pytest.raises(InvalidIrError):
        validate_ir(program, deep=True)
