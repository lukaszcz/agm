"""Tests for `{Hashable K, Eq T}` constraint-block syntax, AST, and scope.

Covers grammar/transformer parsing of a constraint block on every generic
function declaration form (free `def`, `extern def`, `builtin def`, and
builtin/nominal receiver methods), the AST shape it builds, rejection of an
unrecognized constraint name, and the scope-level well-formedness checks: a
constrained name must be an in-scope type parameter (the declaration's own or
its receiver's), no duplicate or redundant (`Eq` + `Hashable`) pairing on one
parameter, and no block on a declaration with no type parameters. A block is
grammatically illegal on `record`/`enum`/`exception`/`type`.

No typing effect is exercised here: constraints are carried on the AST and
validated for well-formedness only.
"""

from __future__ import annotations

import os

import pytest

from agm.agl import PipelineDriver, artifact_serialization
from agm.agl.constraints import ConstraintKind
from agm.agl.parser import AglSyntaxError, parse_program
from agm.agl.scope import AglScopeError, ModuleResolution
from agm.agl.scope.symbols import TypeArgumentsError
from agm.agl.syntax.nodes import Constraint, FuncDef
from agm.agl.syntax.visitor import walk
from tests.agl.module_graph import resolve_inline_entry


def first(source: str) -> FuncDef:
    """Parse *source* and return its first top-level item, asserted a FuncDef."""
    item = parse_program(source.strip()).body.items[0]
    assert isinstance(item, FuncDef)
    return item


def nth(source: str, index: int) -> FuncDef:
    """Parse *source* and return its *index*-th top-level item, asserted a FuncDef."""
    item = parse_program(source.strip()).body.items[index]
    assert isinstance(item, FuncDef)
    return item


def parse_and_resolve(source: str) -> ModuleResolution:
    """Resolve test-only inline source as ``agm exec -c`` does."""
    return resolve_inline_entry(source)


def reject_scope(source: str) -> AglScopeError:
    """Assert that *source* fails scope resolution and return the error."""
    with pytest.raises(AglScopeError) as exc_info:
        parse_and_resolve(source)
    return exc_info.value


# ---------------------------------------------------------------------------
# Grammar / transformer: parsing across every declaration head form
# ---------------------------------------------------------------------------


class TestConstraintBlockParsing:
    def test_free_def_single_constraint(self) -> None:
        fd = first("def f[K, V]{Hashable K}(d: K, v: V) -> K = d")
        assert [(c.kind, c.param) for c in fd.constraints] == [(ConstraintKind.HASHABLE, "K")]

    def test_free_def_multiple_constraints_preserve_order(self) -> None:
        fd = first("def f[K, V]{Hashable K, Eq V}(d: K, v: V) -> K = d")
        assert [(c.kind, c.param) for c in fd.constraints] == [
            (ConstraintKind.HASHABLE, "K"),
            (ConstraintKind.EQ, "V"),
        ]

    def test_constraint_block_allows_trailing_comma(self) -> None:
        fd = first("def f[K]{Hashable K,}(x: K) -> unit = ()")
        assert [(c.kind, c.param) for c in fd.constraints] == [(ConstraintKind.HASHABLE, "K")]

    def test_no_constraint_block_is_empty(self) -> None:
        fd = first("def f[K](x: K) -> K = x")
        assert fd.constraints == ()

    def test_spaced_brace_form_parses(self) -> None:
        fd = first("def f[K] {Hashable K}(x: K) -> K = x")
        assert [(c.kind, c.param) for c in fd.constraints] == [(ConstraintKind.HASHABLE, "K")]

    def test_multiline_block_parses(self) -> None:
        fd = first("def f[K, V]{\n  Hashable K,\n  Eq V,\n}(k: K, v: V) -> K = k")
        assert [(c.kind, c.param) for c in fd.constraints] == [
            (ConstraintKind.HASHABLE, "K"),
            (ConstraintKind.EQ, "V"),
        ]

    def test_extern_def_constraint_block(self) -> None:
        fd = first("extern def f[K]{Hashable K}(x: K) -> unit")
        assert fd.is_extern is True
        assert [(c.kind, c.param) for c in fd.constraints] == [(ConstraintKind.HASHABLE, "K")]

    def test_builtin_def_constraint_block(self) -> None:
        fd = first("builtin def f[K]{Eq K}(x: K) -> unit")
        assert fd.is_builtin is True
        assert [(c.kind, c.param) for c in fd.constraints] == [(ConstraintKind.EQ, "K")]

    def test_program_def_constraint_block(self) -> None:
        fd = first("program def f[K]{Eq K}(x: K) -> unit = ()")
        assert fd.is_program is True
        assert [(c.kind, c.param) for c in fd.constraints] == [(ConstraintKind.EQ, "K")]

    def test_builtin_receiver_method_constrains_receiver_param(self) -> None:
        fd = first("def array[E]::probe{Eq E}(self) -> bool = true")
        assert fd.type_param_slots == ("E",)
        assert [(c.kind, c.param) for c in fd.constraints] == [(ConstraintKind.EQ, "E")]

    def test_builtin_receiver_method_constrains_receiver_and_own_params(self) -> None:
        fd = first("def array[E]::regroup[U]{Eq E, Hashable U}(self, f: E -> U) -> array[U] = self")
        assert fd.type_param_slots == ("E", "U")
        assert [(c.kind, c.param) for c in fd.constraints] == [
            (ConstraintKind.EQ, "E"),
            (ConstraintKind.HASHABLE, "U"),
        ]

    def test_nominal_receiver_method_constrains_receiver_param(self) -> None:
        source = "record Box[T]\n  value: T\ndef Box::get[E]{Eq E}(self) -> E = self.value\n"
        fd = nth(source, 1)
        assert fd.type_param_slots == ("E",)
        assert [(c.kind, c.param) for c in fd.constraints] == [(ConstraintKind.EQ, "E")]

    def test_nominal_receiver_method_constrains_receiver_and_own_params(self) -> None:
        source = (
            "record Box[T]\n"
            "  value: T\n"
            "def Box::m[E, U]{Eq E, Hashable U}(self, other: U) -> bool = true\n"
        )
        fd = nth(source, 1)
        assert fd.type_param_slots == ("E", "U")
        assert [(c.kind, c.param) for c in fd.constraints] == [
            (ConstraintKind.EQ, "E"),
            (ConstraintKind.HASHABLE, "U"),
        ]

    def test_unknown_constraint_name_rejected(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program("def f[K]{Foo K}(x: K) -> K = x")

    def test_type_name_used_as_constraint_name_rejected(self) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program("def f[K]{text K}(x: K) -> K = x")

    @pytest.mark.parametrize(
        "source",
        [
            "record Box[T]{Eq T}\n  value: T\n",
            "enum Outcome[T]{Eq T}\n  | Ok(value: T)\n",
            "exception Failure[T]{Eq T}\n  value: T\n",
            "type Pair[T]{Eq T} = array[T]\n",
        ],
    )
    def test_constraint_block_illegal_on_type_declarations(self, source: str) -> None:
        with pytest.raises(AglSyntaxError):
            parse_program(source)


# ---------------------------------------------------------------------------
# AST node: Constraint
# ---------------------------------------------------------------------------


class TestConstraintNode:
    def test_walk_visits_func_def_constraints(self) -> None:
        fd = first("def f[K, V]{Hashable K, Eq V}(d: K, v: V) -> K = d")
        visited: list[object] = []
        walk(fd, visited.append)
        seen = [(n.kind, n.param) for n in visited if isinstance(n, Constraint)]
        assert seen == [(ConstraintKind.HASHABLE, "K"), (ConstraintKind.EQ, "V")]

    def test_func_def_with_constraints_round_trips_through_artifact_serialization(self) -> None:
        """A cached compiler artifact must be able to restore a constrained FuncDef."""
        fd = first("def f[K, V]{Hashable K, Eq V}(d: K, v: V) -> K = d")
        key = os.urandom(16)
        artifact_serialization.save(key, "test-constraint-round-trip", fd)
        restored = artifact_serialization.load(key, "test-constraint-round-trip")
        assert restored == fd


# ---------------------------------------------------------------------------
# Scope: constraint-block well-formedness
# ---------------------------------------------------------------------------


class TestConstraintScopeValidation:
    def test_valid_single_constraint_accepted(self) -> None:
        parse_and_resolve("def f[K]{Hashable K}(x: K) -> K = x\n()\n")

    def test_valid_multiple_constraints_on_distinct_params_accepted(self) -> None:
        parse_and_resolve("def f[K, V]{Hashable K, Eq V}(k: K, v: V) -> K = k\n()\n")

    def test_same_kind_on_distinct_params_accepted(self) -> None:
        parse_and_resolve("def f[K, V]{Eq K, Eq V}(k: K, v: V) -> K = k\n()\n")

    def test_builtin_receiver_method_constraint_accepted(self) -> None:
        parse_and_resolve("def array[E]::probe{Eq E}(self) -> bool = true\n()\n")

    def test_nominal_receiver_method_constraint_accepted(self) -> None:
        parse_and_resolve(
            "record Box[T]\n  value: T\ndef Box::get[E]{Eq E}(self) -> E = self.value\n()\n"
        )

    def test_constrained_name_not_a_type_parameter_rejected(self) -> None:
        err = reject_scope("def f[K]{Hashable V}(x: K) -> K = x\n()\n")
        span = err.span
        assert span is not None
        # "Hashable V" -- the offending constraint clause, not the whole declaration.
        assert (span.start_line, span.start_col, span.end_line, span.end_col) == (1, 10, 1, 20)

    def test_duplicate_constraint_rejected(self) -> None:
        err = reject_scope("def f[K]{Eq K, Eq K}(x: K) -> K = x\n()\n")
        span = err.span
        assert span is not None
        # The second, repeated "Eq K" clause.
        assert (span.start_line, span.start_col, span.end_line, span.end_col) == (1, 16, 1, 20)

    def test_redundant_implied_pair_eq_then_hashable_rejected(self) -> None:
        reject_scope("def f[K]{Eq K, Hashable K}(x: K) -> K = x\n()\n")

    def test_redundant_implied_pair_hashable_then_eq_rejected(self) -> None:
        reject_scope("def f[K]{Hashable K, Eq K}(x: K) -> K = x\n()\n")

    def test_block_on_declaration_without_type_parameters_rejected(self) -> None:
        err = reject_scope("def f{Eq T}(x: int) -> int = x\n()\n")
        span = err.span
        assert span is not None
        # The first constraint's span, not the whole declaration.
        assert (span.start_line, span.start_col, span.end_line, span.end_col) == (1, 7, 1, 11)

    def test_method_constrained_name_neither_receiver_nor_own_param_rejected(self) -> None:
        """`Z` names neither the receiver's `E` nor a fresh own type parameter."""
        err = reject_scope("def array[E]::m[U]{Eq Z}(self, other: U) -> bool = true\n()\n")
        span = err.span
        assert span is not None
        assert (span.start_line, span.start_col, span.end_line, span.end_col) == (1, 20, 1, 24)

    def test_invalid_receiver_is_rejected_at_its_head_not_its_constraint(self) -> None:
        """An explicit-type-argument receiver on a nominal record is rejected at the receiver.

        Constraint validation must not preempt it with a "not a type
        parameter" error once the receiver's own applied argument covers the
        constrained name.
        """
        err = reject_scope(
            "record Box[T]\n  value: T\ndef Box[T]::get{Eq T}(self) -> T = self.value\n()\n"
        )
        assert isinstance(err, TypeArgumentsError)
        span = err.span
        assert span is not None
        assert (span.start_line, span.start_col, span.end_col) == (3, 5, 11)

    def test_invalid_receiver_reports_receiver_error_not_constraint_error(self) -> None:
        """End to end, the pipeline reports the receiver error, not a constraint one.

        The location pins it to "Box[T]" (the receiver), not "Eq T" (the
        constraint clause, which would be reported at a later column).
        """
        source = (
            "record Box[T]\n"
            "  value: T\n"
            "def Box[T]::get{Eq T}(self) -> T = self.value\n"
            "program def main() -> unit = ()\n"
        )
        prepared = PipelineDriver.prepare_program(source, default_stdlib=False)
        discovery = PipelineDriver().discover_programs(prepared)
        assert discovery.checked is None
        (diagnostic,) = discovery.diagnostics
        assert (diagnostic.line, diagnostic.column, diagnostic.end_column) == (3, 5, 11)


class TestContextualConstraintRecognition:
    def test_user_record_named_eq_coexists_with_constraint_block(self) -> None:
        parse_and_resolve(
            "record Eq\n  value: int\ndef f[K]{Eq K}(x: K) -> K = x\nprint(Eq(value = 1))\n"
        )

    def test_user_record_named_hashable_coexists_with_constraint_block(self) -> None:
        parse_and_resolve(
            "record Hashable\n"
            "  value: int\n"
            "def f[K]{Hashable K}(x: K) -> K = x\n"
            "print(Hashable(value = 1))\n"
        )
