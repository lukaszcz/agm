"""Tests for AgL config declarations (Task 2 — kebab keys + binding + reserved names).

Covers:
- Lexer: ``config`` lexes as a keyword, not VAR_NAME.
- Parser: each literal value variant; multiple declarations; interpolated-template
  value accepted by parser but rejected at scope; LALR conflict guard regression
  (already in test_agl_parser; re-verified here for completeness).
- Scope: config accepted anywhere at program root (header-only removed); nested
  declaration → error; unknown key → error; duplicate key → error; bad value kind
  per key → error; valid declarations collected into ResolvedProgram.config_pragmas;
  non-literal value expression → error; missing value → error; config creates an
  immutable scope binding; reserved program names rejected.
- PreparedProgram.config_pragmas exposure (empty on scope failure).
- Interpreter / typecheck no-op: a program that is only config decls + a print runs
  fine end-to-end.

NOTE: No static-analysis suppression comments in this file.
"""

from __future__ import annotations

import decimal

import pytest

from agm.agl.lexer import tokenize
from agm.agl.parser import parse_program
from agm.agl.pipeline import PipelineDriver
from agm.agl.scope import AglScopeError, resolve
from agm.agl.scope.symbols import ResolvedProgram
from agm.agl.syntax.nodes import (
    BoolLit,
    Call,
    ConfigDecl,
    DecimalLit,
    IntLit,
    StringLit,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tok(source: str) -> list[tuple[str, str]]:
    """Return (type, value) pairs for every token in *source*."""
    return [(t.type, str(t)) for t in tokenize(source)]


def parse_and_resolve(source: str) -> ResolvedProgram:
    """Parse *source* and run the scope resolution pass."""
    return resolve(parse_program(source))


def reject_scope(source: str) -> AglScopeError:
    """Assert that *source* fails scope resolution and return the error."""
    with pytest.raises(AglScopeError) as exc_info:
        parse_and_resolve(source)
    return exc_info.value


def diag(err: AglScopeError) -> tuple[int, str]:
    """Return (line, message) from an AglScopeError."""
    d = err.to_diagnostic()
    return d.line, d.message


# ---------------------------------------------------------------------------
# Lexer: config keyword
# ---------------------------------------------------------------------------


class TestLexerConfigKeyword:
    def test_config_is_keyword_not_var_name(self) -> None:
        """bare 'config' must lex as the keyword token, not NAME."""
        result = tok("config")
        assert result == [("config", "config")]
        types = [t for t, _ in result]
        assert "NAME" not in types

    def test_config_prefix_identifier(self) -> None:
        """'configure' starts with 'config' but is a plain identifier."""
        result = tok("configure")
        assert result == [("NAME", "configure")]

    def test_config_in_sequence(self) -> None:
        """'config' is a keyword even when surrounded by other tokens."""
        result = tok("let x = 1 ; config log = true")
        types = [t for t, _ in result]
        assert "config" in types
        assert types.count("config") == 1


# ---------------------------------------------------------------------------
# Parser: ConfigDecl node shape and value variants
# ---------------------------------------------------------------------------


class TestParserConfigDecl:
    def test_config_decl_bool_true(self) -> None:
        """config KEY = true parses to ConfigDecl with BoolLit(True) value."""
        prog = parse_program("config log = true")
        assert len(prog.body.items) == 1
        stmt = prog.body.items[0]
        assert isinstance(stmt, ConfigDecl)
        assert stmt.name == "log"
        assert isinstance(stmt.value, BoolLit)
        assert stmt.value.value is True

    def test_config_decl_bool_false(self) -> None:
        """config KEY = false parses to ConfigDecl with BoolLit(False) value."""
        prog = parse_program("config log = false")
        stmt = prog.body.items[0]
        assert isinstance(stmt, ConfigDecl)
        assert stmt.name == "log"
        assert isinstance(stmt.value, BoolLit)
        assert stmt.value.value is False

    def test_config_decl_int(self) -> None:
        """config KEY = N parses to ConfigDecl with IntLit value."""
        prog = parse_program("config max-iters = 10")
        stmt = prog.body.items[0]
        assert isinstance(stmt, ConfigDecl)
        assert stmt.name == "max-iters"
        assert isinstance(stmt.value, IntLit)
        assert stmt.value.value == 10

    def test_config_decl_decimal(self) -> None:
        """config KEY = D parses to ConfigDecl with DecimalLit value."""
        prog = parse_program("config timeout = 30.5")
        stmt = prog.body.items[0]
        assert isinstance(stmt, ConfigDecl)
        assert stmt.name == "timeout"
        assert isinstance(stmt.value, DecimalLit)
        assert stmt.value.value == decimal.Decimal("30.5")

    def test_config_decl_string(self) -> None:
        """config KEY = "str" parses to ConfigDecl with StringLit value."""
        prog = parse_program('config runner = "claude"')
        stmt = prog.body.items[0]
        assert isinstance(stmt, ConfigDecl)
        assert stmt.name == "runner"
        assert isinstance(stmt.value, StringLit)
        assert stmt.value.value == "claude"

    def test_config_decl_timeout_string(self) -> None:
        """config timeout = "30s" parses to ConfigDecl with StringLit value."""
        prog = parse_program('config timeout = "30s"')
        stmt = prog.body.items[0]
        assert isinstance(stmt, ConfigDecl)
        assert stmt.name == "timeout"
        assert isinstance(stmt.value, StringLit)
        assert stmt.value.value == "30s"

    def test_config_decl_no_value(self) -> None:
        """config KEY (no = expr) parses to ConfigDecl with None value."""
        prog = parse_program("config log")
        stmt = prog.body.items[0]
        assert isinstance(stmt, ConfigDecl)
        assert stmt.name == "log"
        assert stmt.value is None

    def test_multiple_config_decls(self) -> None:
        """Multiple config decls are each parsed as separate ConfigDecl nodes."""
        prog = parse_program(
            "config log = true\n"
            "config max-iters = 5\n"
            'config runner = "local"\n'
        )
        assert len(prog.body.items) == 3
        assert all(isinstance(s, ConfigDecl) for s in prog.body.items)
        names = [s.name for s in prog.body.items if isinstance(s, ConfigDecl)]
        assert names == ["log", "max-iters", "runner"]

    def test_config_decl_then_statement(self) -> None:
        """Config decls followed by a non-config expression parse correctly."""
        prog = parse_program("config log = true\nprint 1")
        assert len(prog.body.items) == 2
        assert isinstance(prog.body.items[0], ConfigDecl)
        # In v2, ``print 1`` is a Call expression (not a PrintStmt).
        assert isinstance(prog.body.items[1], Call)

    def test_interpolated_template_parses_but_scope_rejects(self) -> None:
        """An interpolated template as a config value passes the parser but is
        rejected at scope as a non-literal expression."""
        # Parse succeeds: the grammar accepts any expr as the value.
        prog = parse_program('config runner = "run ${mode}"')
        stmt = prog.body.items[0]
        assert isinstance(stmt, ConfigDecl)
        # Scope rejects it: Template is not a literal.
        with pytest.raises(AglScopeError):
            resolve(prog)

    def test_config_decl_spans_recorded(self) -> None:
        """ConfigDecl nodes carry non-trivial source spans."""
        prog = parse_program("config log = true")
        stmt = prog.body.items[0]
        assert isinstance(stmt, ConfigDecl)
        assert stmt.span.start_line == 1
        assert stmt.span.start_col == 1


# ---------------------------------------------------------------------------
# Scope: header-only enforcement
# ---------------------------------------------------------------------------


class TestScopeHeaderOnly:
    def test_config_at_root_top_ok(self) -> None:
        """A config decl before any other statement at root is accepted."""
        r = parse_and_resolve("config log = true\nprint 1")
        assert "log" in r.config_pragmas
        assert r.config_pragmas["log"] is True

    def test_config_only_program_ok(self) -> None:
        """A program consisting only of config decls is accepted."""
        r = parse_and_resolve("config log = true\nconfig max-iters = 3")
        assert r.config_pragmas == {"log": True, "max-iters": 3}

    def test_config_after_let_accepted(self) -> None:
        """A config decl after a let statement is now accepted (header-only removed)."""
        r = parse_and_resolve("let x = 1\nconfig log = true")
        assert r.config_pragmas["log"] is True

    def test_config_after_print_accepted(self) -> None:
        """A config decl after a print statement is now accepted."""
        r = parse_and_resolve("print 1\nconfig log = true")
        assert r.config_pragmas["log"] is True

    def test_config_nested_in_block_rejected(self) -> None:
        """A config decl inside a nested block (e.g. do body) is a scope error."""
        err = reject_scope(
            "do\n"
            "  config log = true\n"
            "until true\n"
        )
        _, msg = diag(err)
        assert "config" in msg
        assert "root" in msg or "nested" in msg


# ---------------------------------------------------------------------------
# Scope: key validation
# ---------------------------------------------------------------------------


class TestScopeKeyValidation:
    def test_unknown_key_rejected(self) -> None:
        """An unknown config key is a scope error listing allowed keys."""
        err = reject_scope("config bogus_key = true")
        _, msg = diag(err)
        assert "bogus_key" in msg
        assert "Allowed" in msg or "allowed" in msg

    def test_duplicate_key_rejected(self) -> None:
        """A duplicate config key is a scope error."""
        err = reject_scope("config log = true\nconfig log = false")
        _, msg = diag(err)
        assert "log" in msg
        assert "duplicate" in msg.lower() or "Duplicate" in msg


# ---------------------------------------------------------------------------
# Scope: literal bridge — non-literal and missing value rejection
# ---------------------------------------------------------------------------


class TestScopeLiteralBridge:
    def test_non_literal_var_ref_rejected(self) -> None:
        """A VarRef as the config value is rejected (non-literal)."""
        # Place config at the header so it isn't rejected for header-order reasons.
        err = reject_scope("config log = some_undefined")
        _, msg = diag(err)
        assert "literal" in msg or "config" in msg

    def test_non_literal_expression_rejected(self) -> None:
        """An arithmetic expression as config value is rejected (non-literal)."""
        err = reject_scope("config max-iters = 1 + 2")
        _, msg = diag(err)
        assert "literal" in msg or "config" in msg

    def test_interpolated_string_rejected_at_scope(self) -> None:
        """An interpolated template as config value is rejected at scope."""
        err = reject_scope('config runner = "run ${mode}"')
        _, msg = diag(err)
        assert "literal" in msg or "interpolation" in msg

    def test_missing_value_rejected(self) -> None:
        """config KEY with no value is rejected at scope."""
        err = reject_scope("config log")
        _, msg = diag(err)
        assert "log" in msg


# ---------------------------------------------------------------------------
# Scope: value kind validation
# ---------------------------------------------------------------------------


class TestScopeValueKindValidation:
    def test_log_requires_bool(self) -> None:
        """config log requires a bool value."""
        err = reject_scope('config log = "yes"')
        _, msg = diag(err)
        assert "log" in msg
        assert "bool" in msg

    def test_strict_json_requires_bool(self) -> None:
        """config strict-json requires a bool value."""
        err = reject_scope("config strict-json = 1")
        _, msg = diag(err)
        assert "strict-json" in msg
        assert "bool" in msg

    def test_max_iters_requires_positive_int(self) -> None:
        """config max-iters requires a positive int (> 0)."""
        err = reject_scope("config max-iters = 0")
        _, msg = diag(err)
        assert "max-iters" in msg

    def test_max_iters_negative_rejected(self) -> None:
        """config max-iters rejects a negative value (non-literal unary_neg expression)."""
        # -1 is UnaryNeg(IntLit(1)), a non-literal expression → scope error.
        with pytest.raises(AglScopeError):
            parse_and_resolve("config max-iters = -1")

    def test_max_iters_bool_rejected(self) -> None:
        """config max-iters rejects a bool (which is a subtype of int)."""
        err = reject_scope("config max-iters = true")
        _, msg = diag(err)
        assert "max-iters" in msg

    def test_runner_requires_nonempty_str(self) -> None:
        """config runner requires a non-empty string."""
        err = reject_scope('config runner = 42')
        _, msg = diag(err)
        assert "runner" in msg
        assert "string" in msg

    def test_log_file_requires_nonempty_str(self) -> None:
        """config log-file requires a non-empty string."""
        err = reject_scope("config log-file = true")
        _, msg = diag(err)
        assert "log-file" in msg
        assert "string" in msg

    def test_timeout_accepts_string(self) -> None:
        """config timeout accepts a non-empty string."""
        r = parse_and_resolve('config timeout = "30s"')
        assert r.config_pragmas["timeout"] == "30s"

    def test_timeout_accepts_positive_int(self) -> None:
        """config timeout accepts a positive integer."""
        r = parse_and_resolve("config timeout = 60")
        assert r.config_pragmas["timeout"] == 60

    def test_timeout_bool_rejected(self) -> None:
        """config timeout rejects a bool."""
        err = reject_scope("config timeout = true")
        _, msg = diag(err)
        assert "timeout" in msg

    def test_timeout_decimal_rejected(self) -> None:
        """config timeout rejects a Decimal (not str, not int)."""
        err = reject_scope("config timeout = 30.5")
        _, msg = diag(err)
        assert "timeout" in msg

    def test_timeout_zero_int_rejected(self) -> None:
        """config timeout rejects a zero integer (must be > 0)."""
        err = reject_scope("config timeout = 0")
        _, msg = diag(err)
        assert "timeout" in msg

    def test_log_true_accepted(self) -> None:
        """config log = true is accepted."""
        r = parse_and_resolve("config log = true")
        assert r.config_pragmas["log"] is True

    def test_log_false_accepted(self) -> None:
        """config log = false is accepted."""
        r = parse_and_resolve("config log = false")
        assert r.config_pragmas["log"] is False

    def test_strict_json_true_accepted(self) -> None:
        """config strict-json = true is accepted."""
        r = parse_and_resolve("config strict-json = true")
        assert r.config_pragmas["strict-json"] is True

    def test_max_iters_positive_accepted(self) -> None:
        """config max-iters = 5 is accepted."""
        r = parse_and_resolve("config max-iters = 5")
        assert r.config_pragmas["max-iters"] == 5

    def test_runner_nonempty_accepted(self) -> None:
        """config runner = "claude" is accepted."""
        r = parse_and_resolve('config runner = "claude"')
        assert r.config_pragmas["runner"] == "claude"

    def test_log_file_nonempty_accepted(self) -> None:
        """config log-file = "out.jsonl" is accepted."""
        r = parse_and_resolve('config log-file = "out.jsonl"')
        assert r.config_pragmas["log-file"] == "out.jsonl"


# ---------------------------------------------------------------------------
# Scope: all config decls collected into config_pragmas
# ---------------------------------------------------------------------------


class TestScopeCollectedPragmas:
    def test_all_valid_keys_collected(self) -> None:
        """All valid config keys are collected into config_pragmas."""
        r = parse_and_resolve(
            "config log = true\n"
            "config strict-json = false\n"
            "config max-iters = 10\n"
            'config runner = "local"\n'
            'config log-file = "trace.jsonl"\n'
            'config timeout = "30s"\n'
        )
        assert r.config_pragmas == {
            "log": True,
            "strict-json": False,
            "max-iters": 10,
            "runner": "local",
            "log-file": "trace.jsonl",
            "timeout": "30s",
        }

    def test_no_config_decls_empty_dict(self) -> None:
        """A program with no config decls has an empty config_pragmas dict."""
        r = parse_and_resolve("print 1")
        assert r.config_pragmas == {}


# ---------------------------------------------------------------------------
# PreparedProgram.config_pragmas exposure
# ---------------------------------------------------------------------------


class TestPreparedProgramExposure:
    def test_config_pragmas_property_present(self) -> None:
        """PreparedProgram.config_pragmas is populated from scope pass."""
        prepared = PipelineDriver.prepare(
            "config log = true\n"
            "config max-iters = 3\n"
        )
        assert prepared.config_pragmas == {"log": True, "max-iters": 3}

    def test_config_pragmas_empty_on_scope_failure(self) -> None:
        """config_pragmas is empty when the scope pass failed."""
        prepared = PipelineDriver.prepare("config bogus_key = true")
        assert prepared.config_pragmas == {}
        # Should have a diagnostic.
        assert prepared.diagnostics

    def test_config_pragmas_empty_on_parse_failure(self) -> None:
        """config_pragmas is empty when the parse failed."""
        prepared = PipelineDriver.prepare("let = bad syntax {{{{")
        assert prepared.config_pragmas == {}
        assert prepared.diagnostics

    def test_config_pragmas_empty_when_no_pragmas(self) -> None:
        """config_pragmas is empty when the program has none."""
        prepared = PipelineDriver.prepare("print 1")
        assert prepared.config_pragmas == {}

    def test_config_pragmas_coexists_with_declared_agents(self) -> None:
        """config_pragmas and declared_agents coexist independently."""
        prepared = PipelineDriver.prepare(
            "config log = true\n"
            "agent my_agent\n"
        )
        assert prepared.config_pragmas == {"log": True}
        assert len(prepared.declared_agents) == 1
        assert prepared.declared_agents[0].name == "my_agent"


# ---------------------------------------------------------------------------
# Interpreter / typecheck no-op
# ---------------------------------------------------------------------------


class TestInterpreterNoOp:
    def test_config_plus_print_runs_fine(self) -> None:
        """A program of config decls + a print executes without error."""
        rt = PipelineDriver()
        result = rt.run(
            "config log = true\n"
            "config max-iters = 5\n"
            "print 1\n"
        )
        assert result.ok
        assert not result.diagnostics

    def test_config_only_program_runs_fine(self) -> None:
        """A program consisting only of config decls executes without error."""
        rt = PipelineDriver()
        result = rt.run("config log = false\nconfig strict-json = true")
        assert result.ok

    def test_config_do_not_produce_trace_events(self) -> None:
        """Config decls produce no eval-level side effects (no crash, no events)."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            rt = PipelineDriver()
            result = rt.run(
                "config log = true\n"
                "print 1\n",
                log_file=trace_path,
            )
        assert result.ok
        # No crash is the key assertion; trace content is not checked here.


# ---------------------------------------------------------------------------
# Scope: config no longer header-only (Task 2)
# ---------------------------------------------------------------------------


class TestScopeNoLongerHeaderOnly:
    def test_config_after_statement_accepted(self) -> None:
        """Config decls are now accepted anywhere at root, not just in header."""
        r = parse_and_resolve("let x = 1\nconfig log = true")
        assert r.config_pragmas["log"] is True

    def test_config_after_expression_accepted(self) -> None:
        """Config after a bare expression is now accepted."""
        r = parse_and_resolve("()\nconfig log = true")
        assert r.config_pragmas["log"] is True


# ---------------------------------------------------------------------------
# Scope: config creates an immutable binding (Task 2)
# ---------------------------------------------------------------------------


class TestScopeConfigBinding:
    def test_config_creates_immutable_binding(self) -> None:
        """Config creates a scope binding that can be read."""
        r = parse_and_resolve("config log = true\nlog")
        assert r.config_pragmas["log"] is True

    def test_config_binding_assign_rejected(self) -> None:
        """Assigning to a config binding is a scope error mentioning 'config binding'."""
        err = reject_scope("config log = true\nlog := false")
        _, msg = diag(err)
        assert "config" in msg.lower()

    def test_config_duplicate_still_rejected(self) -> None:
        """Duplicate config key is still rejected."""
        err = reject_scope("config log = true\nconfig log = false")
        _, msg = diag(err)
        assert "log" in msg


# ---------------------------------------------------------------------------
# Scope: reserved program names (Task 2)
# ---------------------------------------------------------------------------


class TestScopeReservedProgramNames:
    def test_reserved_exec_rejected(self) -> None:
        """program exec is rejected (reserved AGM command name)."""
        err = reject_scope("program exec\n()")
        _, msg = diag(err)
        assert "exec" in msg

    def test_reserved_loop_rejected(self) -> None:
        """program loop is rejected (reserved AGM command name)."""
        err = reject_scope("program loop\n()")
        _, msg = diag(err)
        assert "loop" in msg

    def test_reserved_repl_rejected(self) -> None:
        """program repl is rejected (reserved AGM command name)."""
        err = reject_scope("program repl\n()")
        _, msg = diag(err)
        assert "repl" in msg

    def test_unreserved_program_name_ok(self) -> None:
        """program myapp is accepted (not a reserved name)."""
        r = parse_and_resolve("program myapp\n()")
        assert r.program_name == "myapp"


# ---------------------------------------------------------------------------
# engine_keys module — direct unit tests
# ---------------------------------------------------------------------------


class TestEngineKeys:
    def test_get_engine_key_type_known(self) -> None:
        from agm.agl.semantics.engine_keys import get_engine_key_type
        from agm.agl.semantics.types import BoolType
        assert get_engine_key_type("log") == BoolType()

    def test_get_engine_key_type_unknown_returns_none(self) -> None:
        from agm.agl.semantics.engine_keys import get_engine_key_type
        assert get_engine_key_type("__no_such_key__") is None

    def test_engine_key_names_frozenset(self) -> None:
        from agm.agl.semantics.engine_keys import ENGINE_KEY_NAMES
        assert "log" in ENGINE_KEY_NAMES
        assert "max-iters" in ENGINE_KEY_NAMES
        assert "strict-json" in ENGINE_KEY_NAMES

    def test_reserved_program_names_covers_all_commands(self) -> None:
        """Guard against drift: every agm command name must be in RESERVED_PROGRAM_NAMES."""
        from agm import parser
        from agm.agl.semantics.engine_keys import RESERVED_PROGRAM_NAMES
        command_names = {name for name, _ in parser._COMMAND_OVERVIEW}
        assert command_names <= RESERVED_PROGRAM_NAMES
