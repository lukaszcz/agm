"""Nominal constructor keys unify record and enum-member match compilation."""

from __future__ import annotations

import pytest

import agm.agl.matchcompile.compiler as compiler_module
from agm.agl.capabilities import HostCapabilities
from agm.agl.matchcompile.matrix import constructor_inhabits_type
from agm.agl.matchcompile.model import (
    ClosedSignature,
    NominalConstructor,
    Occurrence,
    OccurrenceId,
    RootOccurrenceProvenance,
)
from agm.agl.matchcompile.normalize import MatchCompileInvariantError, signature_for_type
from agm.agl.semantics.type_table import TypeTable
from agm.agl.semantics.types import EnumType, RecordType
from agm.agl.syntax.nodes import Case
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.visitor import walk
from tests.agl.module_graph import resolve_and_check_inline_entry

_CAPS = HostCapabilities(
    supports_shell_exec=True,
    codec_kinds={
        "text": frozenset({"text"}),
        "json": frozenset({"json", "record", "enum", "array", "dict", "int", "decimal", "bool"}),
    },
)


def test_record_and_enum_member_signatures_use_nominal_record_constructors() -> None:
    checked = resolve_and_check_inline_entry(
        "record R(x: int)\nenum E\n  | A(x: int)\nlet e: E = A(x = 1)\ncase e of | A(x = _) => 0",
        _CAPS,
    )
    cases: list[Case] = []
    walk(
        checked.resolved.program,
        lambda node: cases.append(node) if isinstance(node, Case) else None,
    )
    (case,) = cases
    enum_type = checked.node_types[case.subject.node_id]
    assert isinstance(enum_type, EnumType)
    enum_signature = signature_for_type(enum_type, checked.type_env.type_table)
    assert isinstance(enum_signature, ClosedSignature)
    (enum_constructor,) = enum_signature.constructors
    assert isinstance(enum_constructor, NominalConstructor)
    assert isinstance(enum_constructor.record_type, RecordType)

    record_type = checked.type_env.type_table.enum_members(enum_type)[0]
    record_signature = signature_for_type(record_type, checked.type_env.type_table)
    assert record_signature == enum_signature

    with pytest.raises(MatchCompileInvariantError, match="absent from its checked signature"):
        compiler_module._canonical_switch_constructor(
            NominalConstructor(record_type, ()),
            Occurrence(
                OccurrenceId(0),
                0,
                record_type,
                RootOccurrenceProvenance(0, SourceSpan(1, 1, 1, 1, 0, 0)),
            ),
            checked.type_env.type_table,
        )


def test_nominal_constructor_invariants_require_resolved_enum_members() -> None:
    unknown_enum = EnumType("Unknown", decl_id=999)
    constructor = NominalConstructor(RecordType("Member", decl_id=1000), ())

    with pytest.raises(MatchCompileInvariantError, match="cannot resolve enum signature"):
        constructor_inhabits_type(constructor, unknown_enum, TypeTable())
