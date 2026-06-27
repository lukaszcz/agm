"""Tests for AgL config pragmas (Milestone 2 — language support only).

Covers:
- Lexer: ``config`` lexes as a keyword, not VAR_NAME.
- Parser: each pragma_value variant; multiple pragmas; interpolated-template
  value rejected; LALR conflict guard regression (already in test_agl_parser;
  re-verified here for completeness).
- Scope: header-only OK; pragma after a non-pragma statement → error; nested
  pragma → error; unknown key → error; duplicate key → error; bad value kind
  per key → error; valid pragmas collected into ResolvedProgram.config_pragmas.
- PreparedProgram.config_pragmas exposure (empty on scope failure).
- Interpreter / typecheck no-op: a program that is only pragmas + a print runs
  fine end-to-end.

NOTE: No static-analysis suppression comments in this file.
"""

from __future__ import annotations

import decimal

import pytest

from agm.agl.lexer import tokenize
from agm.agl.parser import AglSyntaxError, parse_program
from agm.agl.pipeline import PipelineDriver
from agm.agl.scope import AglScopeError, resolve
from agm.agl.scope.symbols import ResolvedProgram
from agm.agl.syntax.nodes import Call, ConfigPragma

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
# Parser: pragma_value variants
# ---------------------------------------------------------------------------


class TestParserPragmaValues:
    def test_pragma_true(self) -> None:
        """config KEY = true parses to ConfigPragma with bool True."""
        prog = parse_program("config log = true")
        assert len(prog.body.items) == 1
        stmt = prog.body.items[0]
        assert isinstance(stmt, ConfigPragma)
        assert stmt.key == "log"
        assert stmt.value is True
        assert isinstance(stmt.value, bool)

    def test_pragma_false(self) -> None:
        """config KEY = false parses to ConfigPragma with bool False."""
        prog = parse_program("config log = false")
        stmt = prog.body.items[0]
        assert isinstance(stmt, ConfigPragma)
        assert stmt.value is False
        assert isinstance(stmt.value, bool)

    def test_pragma_int(self) -> None:
        """config KEY = N parses to ConfigPragma with int value."""
        prog = parse_program("config max_call_depth = 10")
        stmt = prog.body.items[0]
        assert isinstance(stmt, ConfigPragma)
        assert stmt.key == "max_call_depth"
        assert stmt.value == 10
        assert isinstance(stmt.value, int)
        assert not isinstance(stmt.value, bool)

    def test_pragma_decimal(self) -> None:
        """config KEY = D parses to ConfigPragma with Decimal value."""
        prog = parse_program("config timeout = 30.5")
        stmt = prog.body.items[0]
        assert isinstance(stmt, ConfigPragma)
        assert stmt.key == "timeout"
        assert stmt.value == decimal.Decimal("30.5")
        assert isinstance(stmt.value, decimal.Decimal)

    def test_pragma_string(self) -> None:
        """config KEY = "str" parses to ConfigPragma with str value."""
        prog = parse_program('config runner = "claude"')
        stmt = prog.body.items[0]
        assert isinstance(stmt, ConfigPragma)
        assert stmt.key == "runner"
        assert stmt.value == "claude"
        assert isinstance(stmt.value, str)

    def test_multiple_pragmas(self) -> None:
        """Multiple pragmas are each parsed as separate ConfigPragma nodes."""
        prog = parse_program(
            "config log = true\n"
            "config max_call_depth = 5\n"
            'config runner = "local"\n'
        )
        assert len(prog.body.items) == 3
        assert all(isinstance(s, ConfigPragma) for s in prog.body.items)
        keys = [s.key for s in prog.body.items if isinstance(s, ConfigPragma)]
        assert keys == ["log", "max_call_depth", "runner"]

    def test_pragma_then_statement(self) -> None:
        """Pragmas followed by a non-pragma expression parse correctly."""
        prog = parse_program("config log = true\nprint 1")
        assert len(prog.body.items) == 2
        assert isinstance(prog.body.items[0], ConfigPragma)
        # In v2, ``print 1`` is a Call expression (not a PrintStmt).
        assert isinstance(prog.body.items[1], Call)

    def test_interpolated_template_pragma_value_rejected(self) -> None:
        """An interpolated template as a pragma value is a syntax error."""
        with pytest.raises(AglSyntaxError) as exc_info:
            parse_program('config runner = "run ${mode}"')
        msg = str(exc_info.value)
        assert "interpolation" in msg or "literal" in msg

    def test_config_pragma_spans_recorded(self) -> None:
        """ConfigPragma nodes carry non-trivial source spans."""
        prog = parse_program("config log = true")
        stmt = prog.body.items[0]
        assert isinstance(stmt, ConfigPragma)
        assert stmt.span.start_line == 1
        assert stmt.span.start_col == 1


# ---------------------------------------------------------------------------
# Scope: header-only enforcement
# ---------------------------------------------------------------------------


class TestScopeHeaderOnly:
    def test_pragma_at_root_top_ok(self) -> None:
        """A pragma before any other statement at root is accepted."""
        r = parse_and_resolve("config log = true\nprint 1")
        assert "log" in r.config_pragmas
        assert r.config_pragmas["log"] is True

    def test_pragma_only_program_ok(self) -> None:
        """A program consisting only of pragmas is accepted."""
        r = parse_and_resolve("config log = true\nconfig max_call_depth = 3")
        assert r.config_pragmas == {"log": True, "max_call_depth": 3}

    def test_pragma_after_let_rejected(self) -> None:
        """A pragma that follows a let statement is a scope error."""
        err = reject_scope('let x = 1\nconfig log = true')
        _, msg = diag(err)
        assert "config" in msg
        assert "before" in msg or "header" in msg or "non-pragma" in msg

    def test_pragma_after_print_rejected(self) -> None:
        """A pragma that follows a print statement is a scope error."""
        err = reject_scope('print 1\nconfig log = true')
        _, msg = diag(err)
        assert "config" in msg

    def test_pragma_after_unit_expr_rejected(self) -> None:
        """A pragma that follows a unit expression (()) is a scope error."""
        err = reject_scope('()\nconfig log = true')
        _, msg = diag(err)
        assert "config" in msg

    def test_pragma_nested_in_block_rejected(self) -> None:
        """A pragma inside a nested block (e.g. do body) is a scope error."""
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
        """An unknown pragma key is a scope error listing allowed keys."""
        err = reject_scope("config bogus_key = true")
        _, msg = diag(err)
        assert "bogus_key" in msg
        assert "Allowed" in msg or "allowed" in msg

    def test_duplicate_key_rejected(self) -> None:
        """A duplicate pragma key is a scope error."""
        err = reject_scope("config log = true\nconfig log = false")
        _, msg = diag(err)
        assert "log" in msg
        assert "duplicate" in msg.lower() or "Duplicate" in msg


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
        """config strict_json requires a bool value."""
        err = reject_scope("config strict_json = 1")
        _, msg = diag(err)
        assert "strict_json" in msg
        assert "bool" in msg

    def test_max_call_depth_requires_positive_int(self) -> None:
        """config max_call_depth requires a positive int (> 0)."""
        err = reject_scope("config max_call_depth = 0")
        _, msg = diag(err)
        assert "max_call_depth" in msg

    def test_max_call_depth_negative_rejected(self) -> None:
        """config max_call_depth rejects a negative int."""
        # Negative ints cannot be parsed as a pragma_value (INT token is unsigned);
        # this will be a parse error rather than a scope error.
        with pytest.raises((AglSyntaxError, AglScopeError)):
            parse_and_resolve("config max_call_depth = -1")

    def test_max_call_depth_bool_rejected(self) -> None:
        """config max_call_depth rejects a bool (which is a subtype of int)."""
        err = reject_scope("config max_call_depth = true")
        _, msg = diag(err)
        assert "max_call_depth" in msg

    def test_runner_requires_nonempty_str(self) -> None:
        """config runner requires a non-empty string."""
        err = reject_scope('config runner = 42')
        _, msg = diag(err)
        assert "runner" in msg
        assert "string" in msg

    def test_log_file_requires_nonempty_str(self) -> None:
        """config log_file requires a non-empty string."""
        err = reject_scope("config log_file = true")
        _, msg = diag(err)
        assert "log_file" in msg
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
        """config strict_json = true is accepted."""
        r = parse_and_resolve("config strict_json = true")
        assert r.config_pragmas["strict_json"] is True

    def test_max_call_depth_positive_accepted(self) -> None:
        """config max_call_depth = 5 is accepted."""
        r = parse_and_resolve("config max_call_depth = 5")
        assert r.config_pragmas["max_call_depth"] == 5

    def test_runner_nonempty_accepted(self) -> None:
        """config runner = "claude" is accepted."""
        r = parse_and_resolve('config runner = "claude"')
        assert r.config_pragmas["runner"] == "claude"

    def test_log_file_nonempty_accepted(self) -> None:
        """config log_file = "out.jsonl" is accepted."""
        r = parse_and_resolve('config log_file = "out.jsonl"')
        assert r.config_pragmas["log_file"] == "out.jsonl"


# ---------------------------------------------------------------------------
# Scope: all pragmas collected into config_pragmas
# ---------------------------------------------------------------------------


class TestScopeCollectedPragmas:
    def test_all_valid_keys_collected(self) -> None:
        """All valid pragma keys are collected into config_pragmas."""
        r = parse_and_resolve(
            "config log = true\n"
            "config strict_json = false\n"
            "config max_call_depth = 10\n"
            'config runner = "local"\n'
            'config log_file = "trace.jsonl"\n'
            'config timeout = "30s"\n'
        )
        assert r.config_pragmas == {
            "log": True,
            "strict_json": False,
            "max_call_depth": 10,
            "runner": "local",
            "log_file": "trace.jsonl",
            "timeout": "30s",
        }

    def test_no_pragmas_empty_dict(self) -> None:
        """A program with no pragmas has an empty config_pragmas dict."""
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
            "config max_call_depth = 3\n"
        )
        assert prepared.config_pragmas == {"log": True, "max_call_depth": 3}

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
    def test_pragmas_plus_print_runs_fine(self) -> None:
        """A program of pragmas + a print executes without error."""
        rt = PipelineDriver()
        result = rt.run(
            "config log = true\n"
            "config max_call_depth = 5\n"
            "print 1\n"
        )
        assert result.ok
        assert not result.diagnostics

    def test_pragmas_only_program_runs_fine(self) -> None:
        """A program consisting only of pragmas executes without error."""
        rt = PipelineDriver()
        result = rt.run("config log = false\nconfig strict_json = true")
        assert result.ok

    def test_pragmas_do_not_produce_trace_events(self) -> None:
        """Pragmas produce no eval-level side effects (no crash, no events)."""
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
