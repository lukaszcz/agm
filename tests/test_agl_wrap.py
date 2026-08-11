"""Tests for the pure inline-source ``program def main`` AST transform."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agm.agl.modules.roots import RootSet
from agm.agl.parser import parse_program_seeded, wrap_inline_program
from agm.agl.pipeline import PipelineDriver
from agm.agl.syntax import (
    AssignStmt,
    Block,
    BuiltinVarDecl,
    Call,
    EnumDef,
    ExceptionDef,
    ExportDecl,
    FuncDef,
    ImportDecl,
    InfixDecl,
    LetDecl,
    OpenDecl,
    ParamDecl,
    RecordDef,
    ScopeRegion,
    TypeAlias,
    UnitT,
    VarDecl,
)


def test_wrap_inline_program_partitions_root_items_and_preserves_statement_order() -> None:
    program, next_node_id = parse_program_seeded(
        """\
import helpers
export helpers
open Shared
builtin var setting: int
def helper() -> unit = ()
record Item()
enum State = Ready
exception Problem()
type Count = int
param input: text
infixl |> at 12
scope Shared
def member() -> unit = ()
end Shared
let value = 1
var total = 0
total := value
print value
""",
        start_id=0,
    )
    original_items = program.body.items

    wrapped, transformed_next_node_id = wrap_inline_program(program, next_node_id=next_node_id)

    root_items = wrapped.body.items
    assert root_items[:-1] == original_items[:12]
    assert all(actual is expected for actual, expected in zip(root_items[:-1], original_items[:12]))
    assert all(
        isinstance(
            item,
            (
                ImportDecl,
                ExportDecl,
                FuncDef,
                RecordDef,
                EnumDef,
                ExceptionDef,
                TypeAlias,
                ParamDecl,
                InfixDecl,
                ScopeRegion,
                BuiltinVarDecl,
                OpenDecl,
            ),
        )
        for item in root_items[:-1]
    )

    main = root_items[-1]
    assert isinstance(main, FuncDef)
    assert main.name == "main"
    assert main.is_program is True
    assert main.is_synthetic is True
    assert main.params == ()
    assert main.type_param_slots == ()
    assert isinstance(main.return_type, UnitT)
    assert isinstance(main.body, Block)
    assert main.body.items == original_items[12:]
    assert all(actual is expected for actual, expected in zip(main.body.items, original_items[12:]))
    assert all(isinstance(item, (LetDecl, VarDecl, AssignStmt)) for item in main.body.items[:3])
    assert isinstance(main.body.items[3], Call)
    assert transformed_next_node_id == next_node_id + 3


def test_wrap_inline_program_keeps_scoped_bindings_at_root_and_wraps_local_bindings() -> None:
    program, next_node_id = parse_program_seeded(
        "let Config::answer = 42\nvar Config::count = 0\nlet local = 1\nvar scratch = 0\n",
        start_id=0,
    )
    scoped_let, scoped_var, local_let, local_var = program.body.items

    wrapped, _ = wrap_inline_program(program, next_node_id=next_node_id)

    wrapped_scoped_let, wrapped_scoped_var, main = wrapped.body.items
    assert wrapped_scoped_let is scoped_let
    assert wrapped_scoped_var is scoped_var
    assert isinstance(wrapped_scoped_let, LetDecl)
    assert isinstance(wrapped_scoped_var, VarDecl)
    assert wrapped_scoped_let.scope_path
    assert wrapped_scoped_var.scope_path
    assert isinstance(main, FuncDef)
    assert main.body.items == (local_let, local_var)
    assert all(
        isinstance(item, (LetDecl, VarDecl)) and not item.scope_path for item in main.body.items
    )


def test_wrap_inline_program_keeps_late_exports_with_the_executable_body() -> None:
    program, next_node_id = parse_program_seeded("let value = 1\nexport helpers\n", start_id=0)

    wrapped, _ = wrap_inline_program(program, next_node_id=next_node_id)

    assert len(wrapped.body.items) == 1
    main = wrapped.body.items[0]
    assert isinstance(main, FuncDef)
    assert main.body.items == program.body.items
    assert isinstance(main.body.items[1], ExportDecl)


def test_wrap_inline_program_wraps_an_empty_declaration_only_source() -> None:
    program, next_node_id = parse_program_seeded("def helper() -> unit = ()", start_id=0)

    wrapped, transformed_next_node_id = wrap_inline_program(program, next_node_id=next_node_id)

    (helper, main) = wrapped.body.items
    assert isinstance(helper, FuncDef)
    assert isinstance(main, FuncDef)
    assert main.is_synthetic
    assert isinstance(main.body, Block)
    assert main.body.items == ()
    assert transformed_next_node_id == next_node_id + 3


def test_wrap_inline_program_is_an_identity_when_a_program_def_is_present() -> None:
    program, next_node_id = parse_program_seeded(
        """\
scope Jobs
program def run() -> unit = ()
end Jobs
let value = 1
""",
        start_id=0,
    )

    wrapped, transformed_next_node_id = wrap_inline_program(program, next_node_id=next_node_id)

    assert wrapped is program
    assert transformed_next_node_id == next_node_id


@pytest.mark.parametrize(
    "source",
    (
        "Config::answer\nlet Config::answer = 42",
        "Config::answer\nscope Config\nlet answer = 42\nend Config",
        "Config::answer\nscope Config\nparam answer: int = 42\nend Config",
        "answer\nparam answer: int = 42",
    ),
    ids=("shorthand-let", "region-let", "scoped-param", "root-param"),
)
def test_inline_expression_cannot_reference_a_later_textual_binding(source: str) -> None:
    parsed = PipelineDriver.parse_entry(source)
    assert parsed.program is not None
    wrapped, next_node_id = wrap_inline_program(parsed.program, next_node_id=parsed.next_id)

    prepared = PipelineDriver.prepare_parsed_entry(
        replace(parsed, program=wrapped, next_id=next_node_id),
        roots=RootSet(roots=frozenset()),
        default_stdlib=False,
    )

    assert prepared.resolved is None
    assert len(prepared.diagnostics) == 1
    assert prepared.diagnostics[0].line == 1


def test_wrapped_root_declaration_reference_is_reported_by_the_scope_pipeline() -> None:
    parsed = PipelineDriver.parse_entry("def read() -> int = value\nlet value = 1")
    assert parsed.program is not None
    wrapped, next_node_id = wrap_inline_program(parsed.program, next_node_id=parsed.next_id)

    prepared = PipelineDriver.prepare_parsed_entry(
        replace(parsed, program=wrapped, next_id=next_node_id),
        roots=RootSet(roots=frozenset()),
        default_stdlib=False,
    )

    assert prepared.resolved is None
    assert len(prepared.diagnostics) == 1
    assert "value" in prepared.diagnostics[0].message
