"""Tests for AgL config declarations (lexer, parser, scope, typecheck).

``config KEY = expr`` is a runtime-resolved readable binding (like ``param``).
The scope pass validates placement (root-only), key membership, duplicates, and
creates an immutable ``config_binding``.  Value-type validation happens in the
typecheck pass (an ``Option[T]`` key also accepts a bare inner ``T``).  Runtime
resolution and the CLI/source/config precedence are covered in
``tests/test_agl_config_runtime.py``.

NOTE: No static-analysis suppression comments in this file.
"""

from __future__ import annotations

import decimal
from pathlib import Path

import pytest

from agm.agl.lexer import tokenize
from agm.agl.parser import parse_program
from agm.agl.pipeline import PipelineDriver
from agm.agl.scope import AglScopeError, resolve
from agm.agl.scope.symbols import BinderKind, ResolvedProgram
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


def has_config_binding(r: ResolvedProgram, name: str) -> bool:
    """Return ``True`` when *name* is an immutable config binding in root scope."""
    ref = r.root_scope.bindings.get(name)
    return ref is not None and ref.kind is BinderKind.config_binding


def run_diagnostics(source: str) -> list[str]:
    """Run *source* through the full pipeline and return error messages."""
    result = PipelineDriver().run(source)
    return [d.message for d in result.diagnostics]


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

    def test_config_decl_spans_recorded(self) -> None:
        """ConfigDecl nodes carry non-trivial source spans."""
        prog = parse_program("config log = true")
        stmt = prog.body.items[0]
        assert isinstance(stmt, ConfigDecl)
        assert stmt.span.start_line == 1
        assert stmt.span.start_col == 1


# ---------------------------------------------------------------------------
# Scope: placement (root-only) and binding creation
# ---------------------------------------------------------------------------


class TestScopePlacement:
    def test_config_at_root_top_ok(self) -> None:
        """A config decl before any other statement at root is accepted."""
        r = parse_and_resolve("config log = true\nprint 1")
        assert has_config_binding(r, "log")

    def test_config_only_program_ok(self) -> None:
        """A program consisting only of config decls is accepted."""
        r = parse_and_resolve("config log = true\nconfig max-iters = 3")
        assert has_config_binding(r, "log")
        assert has_config_binding(r, "max-iters")

    def test_config_after_let_accepted(self) -> None:
        """A config decl after a let statement is accepted (config is root-level)."""
        r = parse_and_resolve("let x = 1\nconfig log = true")
        assert has_config_binding(r, "log")

    def test_config_after_print_accepted(self) -> None:
        """A config decl after a print statement is accepted."""
        r = parse_and_resolve("print 1\nconfig log = true")
        assert has_config_binding(r, "log")

    def test_config_after_expression_accepted(self) -> None:
        """Config after a bare expression is accepted."""
        r = parse_and_resolve("()\nconfig log = true")
        assert has_config_binding(r, "log")

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

    def test_undefined_value_name_rejected(self) -> None:
        """A config value referencing an undefined name is a scope error."""
        err = reject_scope("config log = some_undefined")
        _, msg = diag(err)
        assert "some_undefined" in msg


# ---------------------------------------------------------------------------
# Scope: all valid keys accepted as bindings
# ---------------------------------------------------------------------------


class TestScopeAllKeys:
    def test_all_valid_keys_bound(self) -> None:
        """All valid config keys create config bindings."""
        r = parse_and_resolve(
            "config log = true\n"
            "config strict-json = false\n"
            "config max-iters = 10\n"
            'config runner = "local"\n'
            'config log-file = "trace.jsonl"\n'
            'config timeout = "30s"\n'
        )
        for key in ("log", "strict-json", "max-iters", "runner", "log-file", "timeout"):
            assert has_config_binding(r, key)

    def test_no_config_decls_no_binding(self) -> None:
        """A program with no config decls has no config bindings."""
        r = parse_and_resolve("print 1")
        assert not has_config_binding(r, "log")

    def test_bare_config_accepted(self) -> None:
        """A bare ``config KEY`` (no value) is accepted at scope."""
        r = parse_and_resolve("config timeout\nprint 1")
        assert has_config_binding(r, "timeout")


# ---------------------------------------------------------------------------
# Typecheck: value-type validation (no longer a scope-pass concern)
# ---------------------------------------------------------------------------


class TestTypecheckValueValidation:
    def test_log_requires_bool(self) -> None:
        """config log with a text value is a type error."""
        assert run_diagnostics('config log = "yes"')

    def test_strict_json_requires_bool(self) -> None:
        """config strict-json with an int value is a type error."""
        assert run_diagnostics("config strict-json = 1")

    def test_runner_requires_text(self) -> None:
        """config runner with an int value is a type error."""
        assert run_diagnostics("config runner = 42")

    def test_timeout_option_text_rejects_int(self) -> None:
        """config timeout (Option[text]) rejects an int value."""
        assert run_diagnostics("config timeout = 60")

    def test_timeout_accepts_text(self) -> None:
        """config timeout accepts a bare text value (projected into some)."""
        result = PipelineDriver().run('config timeout = "30s"\nprint 1')
        assert result.ok

    def test_log_true_accepted(self) -> None:
        """config log = true type-checks."""
        result = PipelineDriver().run("config log = true\nprint 1", check_only=True)
        assert result.ok

    def test_max_iters_accepted(self) -> None:
        """config max-iters = 5 type-checks."""
        result = PipelineDriver().run("config max-iters = 5\nprint 1")
        assert result.ok


# ---------------------------------------------------------------------------
# Interpreter / runtime no-op
# ---------------------------------------------------------------------------


class TestInterpreterNoOp:
    def test_config_plus_print_runs_fine(self, tmp_path: Path) -> None:
        """A program of config decls + a print executes without error."""
        rt = PipelineDriver()
        result = rt.run(
            "config log = true\n"
            "config max-iters = 5\n"
            "print 1\n",
            log_file=tmp_path / "trace.jsonl",
        )
        assert result.ok
        assert not result.diagnostics

    def test_config_only_program_runs_fine(self) -> None:
        """A program consisting only of config decls executes without error."""
        rt = PipelineDriver()
        result = rt.run("config log = false\nconfig strict-json = true")
        assert result.ok

    def test_config_do_not_crash_with_trace(self) -> None:
        """Config decls run cleanly even with tracing enabled."""
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


# ---------------------------------------------------------------------------
# Scope: config creates an immutable binding
# ---------------------------------------------------------------------------


class TestScopeConfigBinding:
    def test_config_creates_immutable_binding(self) -> None:
        """Config creates a scope binding that can be read."""
        r = parse_and_resolve("config log = true\nlog")
        assert has_config_binding(r, "log")

    def test_config_binding_assign_rejected(self) -> None:
        """Assigning to a config binding is a scope error mentioning 'config'."""
        err = reject_scope("config log = true\nlog := false")
        _, msg = diag(err)
        assert "config" in msg.lower()

    def test_config_duplicate_still_rejected(self) -> None:
        """Duplicate config key is still rejected."""
        err = reject_scope("config log = true\nconfig log = false")
        _, msg = diag(err)
        assert "log" in msg


# ---------------------------------------------------------------------------
# Scope: reserved program names
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
        from agm.agl.semantics.engine_keys import RESERVED_PROGRAM_NAMES
        from agm.command_catalog import COMMAND_NAMES
        assert set(COMMAND_NAMES) <= RESERVED_PROGRAM_NAMES
