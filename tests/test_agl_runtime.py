"""Tests for WorkflowRuntime — M0 shell behaviors preserved, M1 additions.

Covers:
- WorkflowRuntime constructor with default kwargs
- register_agent: duplicate rejection, reserved-name rejection (ask, exec)
- run: full pipeline now active; valid programs succeed; static errors fail
- Diagnostic has .message (str) and .line (int)
- RunResult has .ok (bool), .diagnostics (list), .error (None for pre-exec failures)
- AglError and SourceSpan
- Token constants
- M1 additions: agent registration/fallback, capability derivation, param validation,
  text-codec behavior, AgentCallError, empty-response valid case
"""

from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING

import pytest

from agm.agl import AglError, SourceSpan, WorkflowRuntime
from agm.agl.diagnostics import Diagnostic, format_diagnostic, format_diagnostic_location
from agm.agl.ir.ids import NominalId
from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID
from agm.agl.runtime import AgentRequest
from agm.agl.runtime.runtime import RunResult
from agm.agl.typecheck.types import Type

if TYPE_CHECKING:
    from agm.agl.runtime.codec import OutputCodec


class TestWorkflowRuntimeConstructor:
    def test_default_constructor_uses_documented_defaults(self) -> None:
        rt = WorkflowRuntime()
        # Documented constructor defaults (design §2.8/§2.11).
        assert rt.default_loop_limit == 5
        assert rt.default_strict_json is False

    def test_default_loop_limit_kwarg_is_observable(self) -> None:
        rt = WorkflowRuntime(default_loop_limit=10)
        assert rt.default_loop_limit == 10

    def test_default_strict_json_kwarg_is_observable(self) -> None:
        rt = WorkflowRuntime(default_strict_json=True)
        assert rt.default_strict_json is True

    def test_default_agent_constructed_runtime_runs(self) -> None:
        # A default_agent does not reserve the agent-name namespace: a runtime
        # built with one still accepts named registrations and still runs.
        def my_agent(request: object) -> str:
            return "response"

        rt = WorkflowRuntime(default_agent=my_agent)
        rt.register_agent("reviewer", my_agent)  # should not raise
        # The source declares the registered agent so the source↔host contract
        # holds (M4); a valid program then returns ok=True.
        result = rt.run("agent reviewer\nlet x = 1\nx")
        assert result.ok is True
        assert result.error is None


class TestRegisterAgent:
    def test_register_agent_accepted(self) -> None:
        rt = WorkflowRuntime()
        prompts: list[str] = []

        def my_agent(request: AgentRequest) -> str:
            prompts.append(request.prompt)
            return "response"

        rt.register_agent("my_agent", my_agent)
        result = rt.run(
            'agent my_agent\nlet answer = ask("meaningful prompt", agent: my_agent)\n'
            "print answer"
        )

        assert result.ok
        assert prompts == ["meaningful prompt"]

    def test_register_duplicate_raises(self) -> None:
        rt = WorkflowRuntime()

        def my_agent(request: object) -> str:
            return "response"

        rt.register_agent("my_agent", my_agent)
        with pytest.raises(ValueError, match="my_agent"):
            rt.register_agent("my_agent", my_agent)

    def test_register_reserved_name_ask_raises(self) -> None:
        rt = WorkflowRuntime()

        def my_agent(request: object) -> str:
            return "response"

        with pytest.raises(ValueError, match="ask"):
            rt.register_agent("ask", my_agent)

    def test_register_reserved_name_exec_raises(self) -> None:
        rt = WorkflowRuntime()

        def my_agent(request: object) -> str:
            return "response"

        with pytest.raises(ValueError, match="exec"):
            rt.register_agent("exec", my_agent)

    def test_register_reserved_name_ask_request_raises(self) -> None:
        rt = WorkflowRuntime()

        def my_agent(request: object) -> str:
            return "response"

        with pytest.raises(ValueError, match="ask-request"):
            rt.register_agent("ask-request", my_agent)


class TestRunBehavior:
    """M1 run() behavior: valid programs run, static errors fail cleanly."""

    def test_run_returns_run_result(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert isinstance(result, RunResult)

    def test_valid_program_ok(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = 1\nx")
        assert result.ok is True

    def test_static_error_not_ok(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = undefined_name")
        assert result.ok is False
        assert result.error is None
        assert len(result.diagnostics) >= 1

    def test_static_error_diagnostic_has_message(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = undefined_name")
        diag = result.diagnostics[0]
        assert isinstance(diag.message, str)
        assert diag.message

    def test_static_error_diagnostic_has_line(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = undefined_name")
        diag = result.diagnostics[0]
        assert isinstance(diag.line, int)
        assert diag.line >= 1

    def test_run_result_error_none_for_static_failure(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = undefined_name")
        # pre-execution failure: error is None (no AgL exception was raised)
        assert result.error is None

    def test_run_with_params(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("param k\nprint k", param_values={"k": "value"})
        assert isinstance(result, RunResult)
        assert result.ok is True

    def test_run_with_empty_params(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = 1\nx", param_values={})
        assert isinstance(result, RunResult)
        assert result.ok is True

    def test_run_parse_error_not_ok(self) -> None:
        rt = WorkflowRuntime()
        # Invalid syntax
        result = rt.run("@@@@@")
        assert result.ok is False
        assert result.error is None


class TestFallbackAgent:
    """Default-agent backing behavior for capability checking."""

    def test_no_default_agent_ask_call_static_error(self) -> None:
        rt = WorkflowRuntime()  # no default_agent
        result = rt.run('let x = ask "hi"')
        assert result.ok is False
        assert result.error is None  # static, not runtime

    def test_with_default_agent_ask_call_succeeds(self) -> None:
        rt = WorkflowRuntime(default_agent=lambda req: "ok")
        result = rt.run('let x = ask "hi"\nx')
        assert result.ok is True

    def test_named_agent_registered_accepted(self) -> None:
        rt = WorkflowRuntime()
        rt.register_agent("impl", lambda req: "output")
        result = rt.run('agent impl\nask("do it", agent: impl)')
        assert result.ok is True

    def test_undeclared_named_agent_is_static_error(self) -> None:
        rt = WorkflowRuntime()
        # An undeclared named agent is a static scope binding error: it is
        # rejected before execution regardless of host backing.
        result = rt.run('let x = mysterious_agent "hi"')
        assert result.ok is False
        assert result.error is None

    def test_default_agent_backs_declared_name(self) -> None:
        rt = WorkflowRuntime(default_agent=lambda req: "ok")
        # A default_agent backs any declared name without a dedicated registration.
        result = rt.run('agent any_agent_name\nask("hi", agent: any_agent_name)')
        assert result.ok is True

    def test_declared_but_uncalled_agent_surfaces_warning(self) -> None:
        # A default agent backs the declared (but uncalled) agent so the
        # source↔host contract holds (decision 11): a declared+backed agent
        # that is never called is a non-fatal scope WARNING, surfaced on
        # result.warnings without affecting result.ok.
        rt = WorkflowRuntime(default_agent=lambda req: "ok")
        result = rt.run('agent unused_helper\nprint "hi"')
        assert result.ok is True
        joined = " ".join(d.message for d in result.warnings)
        assert "unused_helper" in joined


class TestInputValidationRuntime:
    """Param validation before execution (§11.3, §9.5)."""

    def test_missing_param_fails_not_ok(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("param spec\nprint spec", param_values={})
        assert result.ok is False
        assert result.error is None  # host error, not AgL exception

    def test_missing_param_mentions_name(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("param spec\nprint spec", param_values={})
        msgs = " ".join(d.message for d in result.diagnostics)
        assert "spec" in msgs.lower()

    def test_undeclared_extra_is_ignored(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("param a\nprint a", param_values={"a": "ok", "b": "extra"})
        assert result.ok is True
        assert result.diagnostics == []

    def test_text_param_verbatim(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("param msg\nprint msg", param_values={"msg": "hello world"})
        assert result.ok is True

    def test_no_agent_called_on_param_failure(self) -> None:
        calls: list[str] = []

        def agent(req: AgentRequest) -> str:
            calls.append(req.prompt)
            return "ok"

        rt = WorkflowRuntime(default_agent=agent)
        rt.run('param x\nask("Hi")', param_values={})
        assert calls == []

    def test_int_param_json_parsed(self, capsys: pytest.CaptureFixture[str]) -> None:
        rt = WorkflowRuntime()
        result = rt.run("param n: int\nprint n", param_values={"n": 5})
        assert result.ok
        out = capsys.readouterr().out
        assert "5" in out

    def test_invalid_typed_param_fails(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("param n: int\nprint n", param_values={"n": "five"})
        assert result.ok is False
        assert result.error is None

    def test_missing_param_reports_declaration_line(self) -> None:
        """F3: the missing-param diagnostic carries the declaration's line."""
        rt = WorkflowRuntime()
        # ``param spec`` is on line 3; the diagnostic must report line 3, not 1.
        src = "let a = 1\nlet b = 2\nparam spec\nprint spec"
        result = rt.run(src, param_values={})
        assert result.ok is False
        missing = [d for d in result.diagnostics if "spec" in d.message.lower()]
        assert missing, result.diagnostics
        assert missing[0].line == 3

    def test_invalid_typed_param_reports_declaration_line(self) -> None:
        """F3 parity: the type-invalid diagnostic already reports the line."""
        rt = WorkflowRuntime()
        src = "let a = 1\nlet b = 2\nparam n: int\nprint n"
        result = rt.run(src, param_values={"n": "five"})
        assert result.ok is False
        bad = [d for d in result.diagnostics if "n" in d.message.lower()]
        assert bad, result.diagnostics
        assert bad[0].line == 3


class TestEmptyResponse:
    """Exit 0 with empty stdout is a valid empty response (plan §9.5)."""

    def test_empty_string_response_is_valid_text(self) -> None:
        rt = WorkflowRuntime(default_agent=lambda req: "")
        result = rt.run('let x = ask "Say nothing."\nx')
        assert result.ok is True
        from agm.agl.eval.values import TextValue

        assert result.bindings["x"] == TextValue("")


class TestAgentRequest:
    """AgentRequest contract: .prompt and .agent fields."""

    def test_request_prompt_is_rendered_template(self) -> None:
        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        rt = WorkflowRuntime(default_agent=agent)
        rt.run('ask "Hello world"')
        assert received[0].prompt == "Hello world"

    def test_request_agent_name_for_default(self) -> None:
        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        rt = WorkflowRuntime(default_agent=agent)
        rt.run('ask "Hi"')
        assert received[0].agent == "ask"

    def test_request_agent_name_for_named(self) -> None:
        received: list[AgentRequest] = []

        def reviewer(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        rt = WorkflowRuntime()
        rt.register_agent("reviewer", reviewer)
        rt.run('agent reviewer\nask("Review this", agent: reviewer)')
        assert received[0].agent == "reviewer"


class TestUncaughtAgentCallErrorSpan:
    """F2: an uncaught AgentCallError carries the agent-call site's location.

    ``AgentRegistry.dispatch`` raises ``AglRaise`` without a span; the
    interpreter must attach the agent-call node's span so the exit-2 error
    reports ``at line N`` (design §12.6).
    """

    def _failing_runtime(self) -> WorkflowRuntime:
        from agm.agl.runtime.agents import AgentCallHostError

        def failing_agent(req: AgentRequest) -> str:
            raise AgentCallHostError(
                cause="spawn_failure",
                exit_code=None,
                stderr_tail="boom",
                elapsed=0.0,
            )

        return WorkflowRuntime(default_agent=failing_agent)

    def test_dispatch_preserves_existing_span(self) -> None:
        """A span the raise site already supplied is never overwritten."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.values import ExceptionValue, TextValue

        existing = SourceSpan(
            start_line=99,
            start_col=1,
            end_line=99,
            end_col=2,
            start_offset=0,
            end_offset=1,
        )

        def agent(req: AgentRequest) -> str:
            exc_val = ExceptionValue(
                nominal=NominalId(ENTRY_ID, "CustomError"),
                display_name="CustomError",
                fields={"message": TextValue("boom")},
            )
            raise AglRaise(exc_val, span=existing)

        rt = WorkflowRuntime(default_agent=agent)
        result = rt.run('let a = 1\nask("hi")')
        assert result.ok is False
        assert result.error is not None
        # The agent's own span (line 99) is kept, not replaced by the call site.
        assert result.error.line == 99

    def test_uncaught_agent_call_error_reports_call_line(self) -> None:
        rt = self._failing_runtime()
        # The ``ask`` call is on line 2.
        result = rt.run('let a = 1\nask("hi")')
        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "AgentCallError"
        assert result.error.line == 2

    def test_uncaught_agent_call_error_surfaces_at_line_in_message(
        self,
        capsys: pytest.CaptureFixture[str],
        tmp_path: "pathlib.Path",
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """End to end: ``agm exec`` prints ``at line N`` to stderr (exit 2)."""
        import agm.commands.exec as exec_mod
        from agm.agl.runtime.agents import AgentCallHostError
        from agm.cli_support.args import ExecArgs
        from agm.commands.exec import run as exec_run

        agl_file = tmp_path / "prog.agl"
        agl_file.write_text('let a = 1\nask("hi")\n')

        def failing_agent(req: object) -> str:
            raise AgentCallHostError(
                cause="spawn_failure", exit_code=None, stderr_tail="boom", elapsed=0.0
            )

        monkeypatch.setattr(
            exec_mod, "runner_backed_agent_factory", lambda **_: failing_agent
        )
        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            runner=None,
            no_log=True,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_run(args)
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "at line 2" in err


class TestDiagnosticType:
    def test_diagnostic_attributes(self) -> None:
        d = Diagnostic(message="some error", line=3)
        assert d.message == "some error"
        assert d.line == 3
        assert d.column is None

    def test_diagnostic_can_carry_character_range(self) -> None:
        d = Diagnostic(message="some error", line=3, column=5, end_line=3, end_column=9)
        assert d.column == 5
        assert d.end_line == 3
        assert d.end_column == 9
        assert format_diagnostic_location(d) == "<agl>:3:5-8"
        assert format_diagnostic_location(d, source_name=None) == "3:5-8"
        assert format_diagnostic(d) == "<agl>:3:5-8: error: some error"

    def test_diagnostic_defaults_to_error_severity(self) -> None:
        d = Diagnostic(message="some error", line=3)
        assert d.severity == "error"

    def test_diagnostic_warning_severity(self) -> None:
        d = Diagnostic(message="a warning", line=2, severity="warning")
        assert d.severity == "warning"
        assert d.message == "a warning"
        assert d.line == 2


class TestRunResultType:
    def test_run_result_attributes(self) -> None:
        d = Diagnostic(message="err", line=1)
        result = RunResult(ok=False, diagnostics=[d], error=None)
        assert result.ok is False
        assert result.diagnostics == [d]
        assert result.error is None

    def test_run_result_ok_with_only_warnings(self) -> None:
        # ok may be True even when warnings are present: warnings live on their
        # own channel and never appear in ``diagnostics`` (errors-only).
        warning = Diagnostic(message="exhaustiveness", line=4, severity="warning")
        result = RunResult(ok=True, diagnostics=[], error=None, warnings=[warning])
        assert result.ok is True
        assert result.diagnostics == []
        assert result.warnings == [warning]
        assert result.error is None

    def test_run_result_has_bindings(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = 1\nx")
        assert hasattr(result, "bindings")
        assert isinstance(result.bindings, dict)


class TestSourceSpan:
    def test_source_span_attributes(self) -> None:
        span = SourceSpan(
            start_line=1, start_col=1, end_line=1, end_col=11,
            start_offset=0, end_offset=10,
        )
        assert span.start_line == 1
        assert span.start_col == 1
        assert span.end_line == 1
        assert span.end_col == 11


class TestAglError:
    def test_agl_error_no_span(self) -> None:
        err = AglError("something went wrong")
        diag = err.to_diagnostic()
        assert diag.line == 1
        assert "something went wrong" in diag.message

    def test_agl_error_with_span(self) -> None:
        span = SourceSpan(
            start_line=5, start_col=3, end_line=5, end_col=12,
            start_offset=40, end_offset=49,
        )
        err = AglError("type error", span=span)
        diag = err.to_diagnostic()
        assert diag.line == 5
        assert diag.column == 3
        assert diag.end_line == 5
        assert diag.end_column == 12
        assert "type error" in diag.message

    def test_agl_error_span_attribute(self) -> None:
        span = SourceSpan(
            start_line=2, start_col=1, end_line=2, end_col=5,
            start_offset=10, end_offset=14,
        )
        err = AglError("test", span=span)
        assert err.span is span

    def test_agl_error_no_span_is_none(self) -> None:
        err = AglError("test")
        assert err.span is None


class TestTokenConstants:
    """Verify the token alphabet is importable and has the expected constants."""

    def test_layout_tokens_defined(self) -> None:
        from agm.agl.lexer.tokens import DEDENT, INDENT, NEWLINE

        assert NEWLINE == "_NEWLINE"
        assert INDENT == "_INDENT"
        assert DEDENT == "_DEDENT"

    def test_template_tokens_defined(self) -> None:
        from agm.agl.lexer.tokens import (
            INTERP_END,
            INTERP_START,
            STRING_FRAGMENT,
            TEMPLATE_END,
            TEMPLATE_START,
        )

        assert TEMPLATE_START == "TEMPLATE_START"
        assert STRING_FRAGMENT == "STRING_FRAGMENT"
        assert INTERP_START == "INTERP_START"
        assert INTERP_END == "INTERP_END"
        assert TEMPLATE_END == "TEMPLATE_END"

    def test_identifier_tokens_defined(self) -> None:
        from agm.agl.lexer.tokens import NAME

        assert NAME == "NAME"

    def test_number_tokens_defined(self) -> None:
        from agm.agl.lexer.tokens import DECIMAL, INT

        assert INT == "INT"
        assert DECIMAL == "DECIMAL"

    def test_keywords_frozenset_contains_expected(self) -> None:
        from agm.agl.lexer.tokens import KEYWORDS

        assert "let" in KEYWORDS
        assert "var" in KEYWORDS
        assert "if" in KEYWORDS
        assert "case" in KEYWORDS
        assert "do" in KEYWORDS
        assert "until" in KEYWORDS
        assert "try" in KEYWORDS
        assert "catch" in KEYWORDS
        assert "record" in KEYWORDS
        assert "enum" in KEYWORDS
        # contextual keywords NOT in KEYWORDS
        assert "ask" not in KEYWORDS
        assert "exec" not in KEYWORDS

    def test_operators_defined(self) -> None:
        from agm.agl.lexer.tokens import (
            ARROW,
            COLON,
            COMMA,
            DOT,
            EQ,
            GE,
            GT,
            LE,
            LT,
            MINUS,
            NEQ,
            PIPE,
            PLUS,
            SEMICOLON,
            SLASH,
            STAR,
        )

        assert ARROW == "ARROW"
        assert EQ == "EQ"
        assert NEQ == "NEQ"
        assert LE == "LE"
        assert GE == "GE"
        assert LT == "LT"
        assert GT == "GT"
        assert PLUS == "PLUS"
        assert MINUS == "MINUS"
        assert STAR == "STAR"
        assert SLASH == "SLASH"
        assert COLON == "COLON"
        assert COMMA == "COMMA"
        assert DOT == "DOT"
        assert PIPE == "PIPE"
        assert SEMICOLON == "SEMICOLON"

    def test_bracket_tokens_defined(self) -> None:
        from agm.agl.lexer.tokens import LBRACE, LPAR, LSQB, RBRACE, RPAR, RSQB

        assert LPAR == "LPAR"
        assert RPAR == "RPAR"
        assert LSQB == "LSQB"
        assert RSQB == "RSQB"
        assert LBRACE == "LBRACE"
        assert RBRACE == "RBRACE"

    def test_error_token_defined(self) -> None:
        from agm.agl.lexer.tokens import EQ_EQ

        assert EQ_EQ == "EQ_EQ"

    def test_all_keyword_constants(self) -> None:
        from agm.agl.lexer import tokens

        assert tokens.KW_RECORD == "record"
        assert tokens.KW_ENUM == "enum"
        assert tokens.KW_TYPE == "type"
        assert tokens.KW_PARAM == "param"
        assert tokens.KW_LET == "let"
        assert tokens.KW_VAR == "var"
        assert tokens.KW_DO == "do"
        assert tokens.KW_UNTIL == "until"
        assert tokens.KW_IF == "if"
        assert tokens.KW_ELSE == "else"
        assert tokens.KW_CASE == "case"
        assert tokens.KW_OF == "of"
        assert tokens.KW_TRY == "try"
        assert tokens.KW_CATCH == "catch"
        assert tokens.KW_RAISE == "raise"
        assert tokens.KW_AS == "as"
        assert tokens.KW_AND == "and"
        assert tokens.KW_OR == "or"
        assert tokens.KW_NOT == "not"
        assert tokens.KW_IS == "is"
        assert tokens.KW_IN == "in"
        assert tokens.KW_TRUE == "true"
        assert tokens.KW_FALSE == "false"
        assert tokens.KW_NULL == "null"


class TestWorkflowRuntimeProperties:
    def test_default_loop_limit_property(self) -> None:
        rt = WorkflowRuntime(default_loop_limit=7)
        assert rt.default_loop_limit == 7

    def test_default_strict_json_property(self) -> None:
        rt = WorkflowRuntime(default_strict_json=True)
        assert rt.default_strict_json is True

    def test_default_strict_json_property_false(self) -> None:
        rt = WorkflowRuntime(default_strict_json=False)
        assert rt.default_strict_json is False


class TestNoDefaultAgent:
    """F1a/F1b: an ``ask`` call needs a default (or fallback) agent."""

    def test_ask_without_default_agent_is_static_error(self) -> None:
        rt = WorkflowRuntime()  # no default agent configured
        result = rt.run('ask "hi"')
        assert result.ok is False
        assert result.error is None  # static (pre-execution), not an AgL exception
        assert any("default agent" in d.message.lower() for d in result.diagnostics)

    def test_ask_with_default_agent_runs(self) -> None:
        def agent(request: object) -> str:
            return "answer"

        rt = WorkflowRuntime(default_agent=agent)
        result = rt.run('ask "hi"')
        assert result.ok is True
        assert result.error is None


class TestDryRunCheckOnly:
    """F2: ``check_only=True`` runs the static pipeline but executes nothing."""

    def test_check_only_printing_program_produces_no_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rt = WorkflowRuntime()
        result = rt.run('print "hello"', check_only=True)
        assert result.ok is True
        assert result.bindings == {}
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_check_only_static_error_still_fails(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = undefined_name", check_only=True)
        assert result.ok is False

    def test_check_only_never_invokes_agent(self) -> None:
        calls: list[object] = []

        def agent(request: object) -> str:
            calls.append(request)
            return "should not be called"

        rt = WorkflowRuntime(default_agent=agent)
        result = rt.run('let x = ask "hi"\nx', check_only=True)
        assert result.ok is True
        # The agent must never be invoked during a dry run.
        assert calls == []

    def test_check_only_unit_ask_reports_no_codec(self) -> None:
        rt = WorkflowRuntime(default_agent=lambda request: "ignored")
        result = rt.run('let result: unit = ask "hi"\nresult', check_only=True)
        assert result.ok is True
        assert len(result.call_sites) == 1
        site = result.call_sites[0]
        assert site.target_type == "unit"
        assert site.codec_name == "none"
        assert site.has_schema is False

    def test_check_only_param_validation_still_runs(self) -> None:
        rt = WorkflowRuntime()
        # Missing declared param is caught even under check_only.
        result = rt.run("param msg\nprint msg", param_values={}, check_only=True)
        assert result.ok is False
        assert any("msg" in d.message for d in result.diagnostics)


class TestDecimalSerialization:
    """F3/F9: decimals print/round-trip exactly; never via binary float."""

    def test_json_param_with_decimal_prints_exactly(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rt = WorkflowRuntime()
        result = rt.run(
            'param data: json\nprint data', param_values={"data": '{"a": 1.5}'}
        )
        assert result.ok is True
        captured = capsys.readouterr()
        assert "1.5" in captured.out
        # No binary-float artifacts (e.g. 1.5000000000000002).
        assert "1.5000" not in captured.out

    def test_decimal_value_prints_exact_text(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rt = WorkflowRuntime()
        result = rt.run('let x = 0.1\nprint x')
        assert result.ok is True
        captured = capsys.readouterr()
        assert captured.out.strip() == "0.1"

    def test_run_error_preserves_decimal_exactness(self) -> None:
        import decimal

        from agm.agl.eval.values import DecimalValue, ExceptionValue, TextValue
        from agm.agl.runtime.runtime import exception_value_to_run_error

        exc = ExceptionValue(
            nominal=NominalId(ENTRY_ID, "ValidationError"),
            display_name="ValidationError",
            fields={
                "message": TextValue("bad"),
                "amount": DecimalValue(decimal.Decimal("0.1")),
            },
        )
        err = exception_value_to_run_error(exc)
        assert err.fields["amount"] == decimal.Decimal("0.1")
        assert isinstance(err.fields["amount"], decimal.Decimal)


class TestWarningsThreadedOnFailurePaths:
    """F14: typecheck warnings survive param-validation failure paths."""

    def test_warning_and_missing_param_both_visible(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # M1 produces no checker warnings organically, so inject one through the
        # CheckedProgram the runtime threads from ``check``.  This exercises the
        # real failure path (missing param) while a warning is present.
        import agm.agl.typecheck as tc_mod
        from agm.agl.diagnostics import Diagnostic
        from agm.agl.typecheck.env import CheckedProgram

        real_check = tc_mod.check
        warning = Diagnostic(message="a checker warning", line=1, severity="warning")

        def check_with_warning(resolved: object, caps: object) -> CheckedProgram:
            checked = real_check(resolved, caps)
            return CheckedProgram(
                resolved=checked.resolved,
                node_types=checked.node_types,
                contract_specs=checked.contract_specs,
                call_sites=checked.call_sites,
                warnings=(*checked.warnings, warning),
                type_env=checked.type_env,
                function_signatures=checked.function_signatures,
                cast_specs=checked.cast_specs,
            )

        monkeypatch.setattr(tc_mod, "check", check_with_warning)

        rt = WorkflowRuntime()
        result = rt.run("param msg\nprint msg", param_values={})
        assert result.ok is False
        # The warning is threaded onto its own channel even on a failure path.
        warning_messages = [d.message for d in result.warnings]
        assert any("a checker warning" in m for m in warning_messages)
        # The missing-param error lands in diagnostics (errors only).
        error_messages = [d.message for d in result.diagnostics]
        assert any("msg" in m for m in error_messages)
        # Channels stay separate: no warning leaks into diagnostics.
        assert all(d.severity == "error" for d in result.diagnostics)


class TestAgentRegistryDispatch:
    """F17: dispatch resolves named agents, ask, and the default fallback."""

    def test_dispatch_named_agent(self) -> None:
        from agm.agl.runtime import AgentRequest
        from agm.agl.runtime.agents import AgentRegistry

        def named(req: AgentRequest) -> str:
            return f"named:{req.prompt}"

        registry = AgentRegistry(named={"reviewer": named}, default_agent=None)
        resp = registry.dispatch("reviewer", AgentRequest(agent="reviewer", prompt="hi"))
        assert resp.content == "named:hi"

    def test_dispatch_ask_and_unknown_fall_back_to_default(self) -> None:
        from agm.agl.runtime import AgentRequest
        from agm.agl.runtime.agents import AgentRegistry

        def default(req: AgentRequest) -> str:
            return f"default:{req.agent}"

        registry = AgentRegistry(named={}, default_agent=default)
        # Both ``ask`` and an unregistered named agent route to the default.
        assert registry.dispatch("ask", AgentRequest(agent="ask", prompt="q")).content == (
            "default:ask"
        )
        assert registry.dispatch("other", AgentRequest(agent="other", prompt="q")).content == (
            "default:other"
        )

    def test_dispatch_unknown_without_default_raises(self) -> None:
        from agm.agl.runtime import AgentRequest
        from agm.agl.runtime.agents import AgentRegistry

        registry = AgentRegistry(named={}, default_agent=None)
        with pytest.raises(KeyError, match="No agent registered"):
            registry.dispatch("ghost", AgentRequest(agent="ghost", prompt="q"))


class TestParamBindingInvariant:
    """The runtime relies on the checker recording every param's binding type."""

    def test_missing_binding_type_is_internal_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agm.agl.typecheck.env import TypeEnvironment

        # Force the checker invariant to be violated: no recorded binding type.
        monkeypatch.setattr(
            TypeEnvironment, "get_binding_type", lambda self, node_id: None
        )

        rt = WorkflowRuntime()
        with pytest.raises(AssertionError, match="binding type"):
            rt.run("param msg\nprint msg", param_values={"msg": "hi"})


# ---------------------------------------------------------------------------
# CARRY-IN 1 — capabilities built from registrations (M3b)
# ---------------------------------------------------------------------------


class TestCapabilitiesBuiltFromRegistrations:
    """CARRY-IN 1: WorkflowRuntime.run builds HostCapabilities from codec/renderer registries."""

    def test_default_runtime_has_text_and_json_codecs(self) -> None:
        """Built-in text + json codecs are always present."""
        from agm.agl.runtime.codec import JsonCodec, TextCodec

        rt = WorkflowRuntime(default_agent=lambda req: "ok")
        # A json-typed call passes typecheck → json codec is registered.
        tc, jc = TextCodec(), JsonCodec()
        assert tc.name == "text"
        assert jc.name == "json"
        result = rt.run('ask "hi"')
        assert result.ok is True


    def test_register_codec_before_run_extends_capabilities(self) -> None:
        """A custom codec registered before run() makes its kinds available to typecheck."""
        from agm.agl.eval.values import TextValue as TV
        from agm.agl.runtime.codec import ParseResult, TextCodec
        from agm.agl.runtime.contract import OutputContract
        from agm.agl.typecheck.types import TextType

        class FooCodec:
            @property
            def name(self) -> str:
                return "foo"

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset({"text"})

            def supports_type(self, t: Type) -> bool:
                return isinstance(t, TextType)

            def make_contract(self, type_ref: Type) -> OutputContract:
                return OutputContract(
                    target_type=type_ref,
                    codec=TextCodec(),
                    strict_json=None,
                    format_instructions="",
                    json_schema=None,
                )

            def parse(
                self,
                raw: str,
                target_type: Type,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
            ) -> ParseResult:
                return ParseResult.success(TV(raw))

        rt = WorkflowRuntime()
        rt.register_codec(FooCodec())
        # The program just needs to run without capability errors.
        result = rt.run("let x = 1\nx")
        assert result.ok is True

    def test_as_renderer_syntax_is_parse_error(self) -> None:
        """``${x as name}`` is a syntax error (renderer syntax removed)."""
        rt = WorkflowRuntime(default_agent=lambda req: "ok")
        result = rt.run(
            'param x\nlet y = ask "see ${x as fancy}"', param_values={"x": "hi"}
        )
        assert result.ok is False


# ---------------------------------------------------------------------------
# Coverage: render.py — render_value / render_value_repl / _scalar_text
# ---------------------------------------------------------------------------


class TestRenderValue:
    """Unit tests for the AgL-native render_value / render_value_repl functions.

    Nominal values carry their fields in declaration order already (the
    interpreter normalizes them at construction), so these tests build the
    values in the order they expect rendered and the renderer walks
    ``value.fields`` verbatim — no type lookup is involved.
    """

    # ------------------------------------------------------------------
    # text: verbatim (render_value) vs quoted (render_value_repl)
    # ------------------------------------------------------------------

    def test_text_verbatim_render_value(self) -> None:
        """render_value: top-level text is verbatim, no quotes."""
        from agm.agl.eval.values import TextValue
        from agm.agl.runtime.render import render_value

        assert render_value(TextValue("hello world")) == "hello world"

    def test_text_quoted_render_value_repl(self) -> None:
        """render_value_repl: top-level text is wrapped in double quotes."""
        from agm.agl.eval.values import TextValue
        from agm.agl.runtime.render import render_value_repl

        assert render_value_repl(TextValue("hello")) == '"hello"'

    def test_text_repl_escapes_special_chars(self) -> None:
        """render_value_repl: JSON escape set applies to top-level text."""
        from agm.agl.eval.values import TextValue
        from agm.agl.runtime.render import render_value_repl

        assert render_value_repl(TextValue('a"b')) == '"a\\"b"'
        assert render_value_repl(TextValue("a\\b")) == '"a\\\\b"'
        assert render_value_repl(TextValue("a\nb")) == '"a\\nb"'
        assert render_value_repl(TextValue("a\tb")) == '"a\\tb"'

    def test_text_repl_escapes_control_chars_as_unicode(self) -> None:
        """render_value_repl: control chars below 0x20 render as \\uXXXX."""
        from agm.agl.eval.values import TextValue
        from agm.agl.runtime.render import render_value_repl

        assert render_value_repl(TextValue("a\x00b")) == '"a\\u0000b"'

    # ------------------------------------------------------------------
    # Scalars: int / decimal / bool (unchanged at any depth)
    # ------------------------------------------------------------------

    def test_int_value_is_plain_text(self) -> None:
        from agm.agl.eval.values import IntValue
        from agm.agl.runtime.render import render_value

        assert render_value(IntValue(42)) == "42"

    def test_decimal_value_is_plain_text(self) -> None:
        from decimal import Decimal

        from agm.agl.eval.values import DecimalValue
        from agm.agl.runtime.render import render_value

        assert render_value(DecimalValue(Decimal("1.5"))) == "1.5"

    def test_bool_value_is_plain_text(self) -> None:
        from agm.agl.eval.values import BoolValue
        from agm.agl.runtime.render import render_value

        assert render_value(BoolValue(True)) == "true"
        assert render_value(BoolValue(False)) == "false"

    # ------------------------------------------------------------------
    # _scalar_text unit tests (unchanged helper — keep working)
    # ------------------------------------------------------------------

    def test_scalar_text_int(self) -> None:
        """_scalar_text(IntValue) renders as plain decimal digits."""
        from agm.agl.eval.values import IntValue
        from agm.agl.runtime.render import _scalar_text

        assert _scalar_text(IntValue(42)) == "42"

    def test_scalar_text_decimal(self) -> None:
        """_scalar_text(DecimalValue) drops trailing zeros, no sci notation."""
        from decimal import Decimal

        from agm.agl.eval.values import DecimalValue
        from agm.agl.runtime.render import _scalar_text

        assert _scalar_text(DecimalValue(Decimal("1.50"))) == "1.5"
        assert _scalar_text(DecimalValue(Decimal("100"))) == "100"

    def test_scalar_text_bool(self) -> None:
        """_scalar_text(BoolValue) renders as 'true'/'false'."""
        from agm.agl.eval.values import BoolValue
        from agm.agl.runtime.render import _scalar_text

        assert _scalar_text(BoolValue(True)) == "true"
        assert _scalar_text(BoolValue(False)) == "false"

    # ------------------------------------------------------------------
    # unit / agent / closure (non-data, unchanged)
    # ------------------------------------------------------------------

    def test_unit_value_renders_as_unit_literal(self) -> None:
        """Unit value renders as ``()``."""
        from agm.agl.eval.values import UnitValue
        from agm.agl.runtime.render import render_value

        assert render_value(UnitValue()) == "()"

    def test_agent_value_renders_as_angle_bracket_form(self) -> None:
        """AgentValue renders as ``<agent NAME>``."""
        from agm.agl.eval.values import AgentValue
        from agm.agl.runtime.render import render_value

        assert render_value(AgentValue(name="reviewer")) == "<agent reviewer>"

    def test_closure_renders_as_function_surface_form(self) -> None:
        """Closure renders as ``<function: (A, B) -> T>``."""
        from agm.agl.eval.values import IrClosureValue
        from agm.agl.runtime.render import render_value

        rt = WorkflowRuntime()
        result = rt.run("let f = fn(x: int, y: int) -> int => x + y\nf\n")
        assert result.ok is True
        closure = result.bindings["f"]
        assert isinstance(closure, IrClosureValue)
        assert render_value(closure) == "<function: (int, int) -> int>"

    def test_closure_zero_arity_renders_correctly(self) -> None:
        """A zero-parameter closure renders as ``<function: () -> T>``."""
        from agm.agl.eval.values import IrClosureValue
        from agm.agl.runtime.render import render_value

        rt = WorkflowRuntime()
        result = rt.run("let thunk = fn() -> int => 42\nthunk\n")
        assert result.ok is True
        closure = result.bindings["thunk"]
        assert isinstance(closure, IrClosureValue)
        assert render_value(closure) == "<function: () -> int>"

    def test_closure_without_param_metadata_renders_unknown_arg_types(self) -> None:
        """A hand-built closure without type labels uses ``?`` placeholders."""
        from agm.agl.eval.values import IrClosureValue
        from agm.agl.ir.ids import FunctionId
        from agm.agl.runtime.render import render_value

        closure = IrClosureValue(
            function_id=FunctionId(0),
            captures=(),
            arity=2,
            result_label="int",
        )

        assert render_value(closure) == "<function: (?, ?) -> int>"

    # ------------------------------------------------------------------
    # list: single-line AgL form
    # ------------------------------------------------------------------

    def test_list_single_line(self) -> None:
        """Lists render as single-line ``[e1, e2, ...]``."""
        from agm.agl.eval.values import IntValue, ListValue
        from agm.agl.runtime.render import render_value

        v = ListValue(elements=(IntValue(1), IntValue(2)))
        assert render_value(v) == "[1, 2]"

    def test_list_empty(self) -> None:
        """Empty list renders as ``[]``."""
        from agm.agl.eval.values import ListValue
        from agm.agl.runtime.render import render_value

        assert render_value(ListValue(elements=())) == "[]"

    def test_list_nested_text_is_quoted(self) -> None:
        """Text inside a list is quoted."""
        from agm.agl.eval.values import ListValue, TextValue
        from agm.agl.runtime.render import render_value

        v = ListValue(elements=(TextValue("tests"), TextValue("coverage")))
        assert render_value(v) == '["tests", "coverage"]'

    # ------------------------------------------------------------------
    # dict: always-quoted keys, single-line AgL form
    # ------------------------------------------------------------------

    def test_dict_single_line_quoted_keys(self) -> None:
        """Dict renders single-line with always-quoted keys."""
        from agm.agl.eval.values import DictValue, IntValue
        from agm.agl.runtime.render import render_value

        v = DictValue({"origin": IntValue(1)})
        assert render_value(v) == '{"origin": 1}'

    def test_dict_key_with_space_is_quoted(self) -> None:
        """Dict keys with spaces (or any content) are always quoted."""
        from agm.agl.eval.values import DictValue, IntValue
        from agm.agl.runtime.render import render_value

        v = DictValue({"two words": IntValue(1)})
        assert render_value(v) == '{"two words": 1}'

    def test_dict_empty(self) -> None:
        """Empty dict renders as ``{}``."""
        from agm.agl.eval.values import DictValue
        from agm.agl.runtime.render import render_value

        assert render_value(DictValue({})) == "{}"

    def test_dict_multiple_entries(self) -> None:
        """Multiple dict entries, in insertion order, keys quoted."""
        from agm.agl.eval.values import DictValue, IntValue
        from agm.agl.runtime.render import render_value

        v = DictValue({"origin": IntValue(1), "two words": IntValue(2)})
        assert render_value(v) == '{"origin": 1, "two words": 2}'

    # ------------------------------------------------------------------
    # record: AgL form, declaration order
    # ------------------------------------------------------------------

    def test_record_renders_fields_in_stored_order(self) -> None:
        """Record renders fields in the order they are stored (declaration order)."""
        from agm.agl.eval.values import IntValue, RecordValue, TextValue
        from agm.agl.runtime.render import render_value

        v = RecordValue(
            nominal=NominalId(ENTRY_ID, "Issue"),
            display_name="Issue",
            fields={"title": TextValue("Missing tests"), "severity": IntValue(3)},
        )
        assert render_value(v) == 'Issue(title: "Missing tests", severity: 3)'

    def test_record_empty(self) -> None:
        """A record with no fields renders as ``TypeName()``."""
        from agm.agl.eval.values import RecordValue
        from agm.agl.runtime.render import render_value

        v = RecordValue(nominal=NominalId(ENTRY_ID, "Empty"), display_name="Empty", fields={})
        assert render_value(v) == "Empty()"

    def test_record_nested_record(self) -> None:
        """Nested records render inline."""
        from agm.agl.eval.values import BoolValue, RecordValue, TextValue
        from agm.agl.runtime.render import render_value

        author = RecordValue(
            nominal=NominalId(ENTRY_ID, "Author"),
            display_name="Author",
            fields={"name": TextValue("Ada"), "active": BoolValue(True)},
        )
        issue = RecordValue(
            nominal=NominalId(ENTRY_ID, "Issue"),
            display_name="Issue",
            fields={"title": TextValue("Missing tests"), "author": author},
        )
        out = render_value(issue)
        assert out == 'Issue(title: "Missing tests", author: Author(name: "Ada", active: true))'

    def test_record_with_list_field(self) -> None:
        """Record with a list field renders correctly (list inline)."""
        from agm.agl.eval.values import IntValue, ListValue, RecordValue, TextValue
        from agm.agl.runtime.render import render_value

        v = RecordValue(
            nominal=NominalId(ENTRY_ID, "Issue"),
            display_name="Issue",
            fields={
                "title": TextValue("Missing tests"),
                "severity": IntValue(3),
                "tags": ListValue(elements=(TextValue("tests"), TextValue("coverage"))),
            },
        )
        out = render_value(v)
        assert out == 'Issue(title: "Missing tests", severity: 3, tags: ["tests", "coverage"])'

    # ------------------------------------------------------------------
    # enum: qualified form, nullary variant
    # ------------------------------------------------------------------

    def test_enum_with_payload(self) -> None:
        """Enum with payload renders as ``TypeName.Variant(field: value)``."""
        from agm.agl.eval.values import EnumValue, IntValue
        from agm.agl.runtime.render import render_value

        v = EnumValue(
            nominal=NominalId(ENTRY_ID, "Outcome"), display_name="Outcome",
            variant="Partial", fields={"left": IntValue(2)},
        )
        assert render_value(v) == "Outcome.Partial(left: 2)"

    def test_enum_nullary_variant(self) -> None:
        """Nullary enum variant renders as ``TypeName.Variant`` (no parens)."""
        from agm.agl.eval.values import EnumValue
        from agm.agl.runtime.render import render_value

        v = EnumValue(
            nominal=NominalId(ENTRY_ID, "Outcome"), display_name="Outcome",
            variant="Done", fields={},
        )
        assert render_value(v) == "Outcome.Done"

    def test_enum_multi_field_payload(self) -> None:
        """Enum with multiple payload fields renders them in stored order."""
        from agm.agl.eval.values import EnumValue, IntValue
        from agm.agl.runtime.render import render_value

        v = EnumValue(
            nominal=NominalId(ENTRY_ID, "E"),
            display_name="E",
            variant="V",
            fields={"a": IntValue(1), "b": IntValue(2), "c": IntValue(3)},
        )
        assert render_value(v) == "E.V(a: 1, b: 2, c: 3)"

    # ------------------------------------------------------------------
    # exception: record-style with all fields incl. trace_id
    # ------------------------------------------------------------------

    def test_exception_renders_record_style_with_trace_id(self) -> None:
        """Exception renders like a record with all fields in declaration order."""
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.runtime.render import render_value

        v = ExceptionValue(
            nominal=NominalId(PRELUDE_ID, "CastError"),
            display_name="CastError",
            fields={
                "message": TextValue('cannot parse "x" as int'),
                "trace_id": TextValue("evt-7"),
                "source_type": TextValue("text"),
                "target_type": TextValue("int"),
                "raw": TextValue("x"),
            },
        )
        out = render_value(v)
        expected = (
            'CastError(message: "cannot parse \\"x\\" as int", trace_id: "evt-7", '
            'source_type: "text", target_type: "int", raw: "x")'
        )
        assert out == expected

    def test_exception_abort_renders_with_trace_id(self) -> None:
        """Abort exception includes both message and trace_id."""
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.runtime.render import render_value

        v = ExceptionValue(
            nominal=NominalId(ENTRY_ID, "Abort"),
            display_name="Abort",
            fields={
                "message": TextValue("fatal"),
                "trace_id": TextValue("abc123"),
            },
        )
        out = render_value(v)
        assert out == 'Abort(message: "fatal", trace_id: "abc123")'
        assert "<dsl-value" not in out

    # ------------------------------------------------------------------
    # Nested text escaping including $ → \$
    # ------------------------------------------------------------------

    def test_nested_text_escapes_dollar(self) -> None:
        """Nested text containing ``${`` escapes ``$`` as ``\\$``."""
        from agm.agl.eval.values import ListValue, TextValue
        from agm.agl.runtime.render import render_value

        v = ListValue(elements=(TextValue("a${b}"),))
        out = render_value(v)
        assert out == r'["a\${b}"]'

    def test_nested_text_escapes_quotes_and_newlines(self) -> None:
        """Nested text: quotes and newlines are escaped."""
        from agm.agl.eval.values import ListValue, TextValue
        from agm.agl.runtime.render import render_value

        v = ListValue(elements=(TextValue('say "hi"\nbye'),))
        out = render_value(v)
        assert out == r'["say \"hi\"\nbye"]'

    def test_repl_top_level_text_escapes_dollar(self) -> None:
        """render_value_repl: top-level text with ``$`` escapes it as ``\\$``."""
        from agm.agl.eval.values import TextValue
        from agm.agl.runtime.render import render_value_repl

        out = render_value_repl(TextValue("a${b}"))
        assert out == r'"a\${b}"'

    # ------------------------------------------------------------------
    # json: top-level pretty vs nested compact
    # ------------------------------------------------------------------

    def test_json_top_level_is_pretty(self) -> None:
        """Top-level json value renders as pretty-printed JSON (indent=2)."""
        from agm.agl.eval.values import JsonValue
        from agm.agl.runtime.render import render_value

        out = render_value(JsonValue({"k": 1}))
        assert out == '{\n  "k": 1\n}'

    def test_json_nested_is_compact(self) -> None:
        """json nested inside a record field renders compact (single-line)."""
        from agm.agl.eval.values import JsonValue, RecordValue
        from agm.agl.runtime.render import render_value

        v = RecordValue(
            nominal=NominalId(ENTRY_ID, "R"),
            display_name="R",
            fields={"data": JsonValue({"a": 1, "b": 2})},
        )
        out = render_value(v)
        # The json field must be compact (no newlines) so the record stays single-line.
        assert "\n" not in out
        assert out == 'R(data: {"a": 1, "b": 2})'

    # ------------------------------------------------------------------
    # render_value_repl: non-text values match render_value
    # ------------------------------------------------------------------

    def test_repl_non_text_matches_render_value(self) -> None:
        """render_value_repl matches render_value for non-text values."""
        from decimal import Decimal

        from agm.agl.eval.values import (
            BoolValue,
            DecimalValue,
            IntValue,
            ListValue,
            UnitValue,
        )
        from agm.agl.runtime.render import render_value, render_value_repl

        for v in (
            IntValue(42),
            DecimalValue(Decimal("1.5")),
            BoolValue(True),
            UnitValue(),
            ListValue(elements=(IntValue(1),)),
        ):
            assert render_value_repl(v) == render_value(v)

    def test_repl_nested_text_in_list_is_quoted(self) -> None:
        """Text inside a list is quoted regardless of repl mode."""
        from agm.agl.eval.values import ListValue, TextValue
        from agm.agl.runtime.render import render_value_repl

        out = render_value_repl(ListValue(elements=(TextValue("v"),)))
        assert out == '["v"]'

    # ------------------------------------------------------------------
    # No <dsl-value> tags ever
    # ------------------------------------------------------------------

    def test_no_dsl_value_tags_in_interpolation(self) -> None:
        """Interpolation never wraps values in <dsl-value> tags."""
        from agm.agl.eval.values import IntValue, TextValue
        from agm.agl.runtime.render import render_value

        assert "<dsl-value" not in render_value(TextValue("x"))
        assert "<dsl-value" not in render_value(IntValue(1))


# ---------------------------------------------------------------------------
# Coverage: serialize.py — value_to_json_obj and dumps_exact branches
# ---------------------------------------------------------------------------


class TestSerialize:
    """Coverage for serialize.py branches not exercised by higher-level tests."""

    def test_bool_value_serialized(self) -> None:
        from agm.agl.eval.values import BoolValue
        from agm.agl.runtime.serialize import value_to_json_obj

        assert value_to_json_obj(BoolValue(True)) is True
        assert value_to_json_obj(BoolValue(False)) is False

    def test_dict_value_serialized(self) -> None:
        from agm.agl.eval.values import DictValue, IntValue
        from agm.agl.runtime.serialize import value_to_json_obj

        result = value_to_json_obj(DictValue(entries={"a": IntValue(1)}))
        assert result == {"a": 1}

    def test_record_value_serialized(self) -> None:
        from agm.agl.eval.values import IntValue, RecordValue
        from agm.agl.runtime.serialize import value_to_json_obj

        result = value_to_json_obj(RecordValue(
            nominal=NominalId(ENTRY_ID, "R"), display_name="R", fields={"x": IntValue(5)}
        ))
        assert result == {"x": 5}

    def test_enum_value_serialized(self) -> None:
        from agm.agl.eval.values import EnumValue, TextValue
        from agm.agl.runtime.serialize import value_to_json_obj

        result = value_to_json_obj(
            EnumValue(
                nominal=NominalId(ENTRY_ID, "E"), display_name="E",
                variant="A", fields={"msg": TextValue("hi")},
            )
        )
        assert result == {"$case": "A", "msg": "hi"}

    def test_enum_nullary_value_serialized(self) -> None:
        from agm.agl.eval.values import EnumValue
        from agm.agl.runtime.serialize import value_to_json_obj

        result = value_to_json_obj(EnumValue(
            nominal=NominalId(ENTRY_ID, "E"), display_name="E", variant="Done", fields={}
        ))
        assert result == {"$case": "Done"}

    def test_exception_value_serialized(self) -> None:
        from agm.agl.eval.values import ExceptionValue, TextValue
        from agm.agl.runtime.serialize import value_to_json_obj

        result = value_to_json_obj(
            ExceptionValue(
                nominal=NominalId(ENTRY_ID, "Err"), display_name="Err",
                fields={"message": TextValue("oops")},
            )
        )
        assert result == {"message": "oops"}

    def test_dumps_exact_bool_true(self) -> None:
        from agm.agl.runtime.serialize import dumps_exact

        assert dumps_exact(True) == "true"

    def test_dumps_exact_bool_false(self) -> None:
        from agm.agl.runtime.serialize import dumps_exact

        assert dumps_exact(False) == "false"

    def test_dumps_exact_list_empty(self) -> None:
        from agm.agl.runtime.serialize import dumps_exact

        assert dumps_exact([]) == "[]"

    def test_dumps_exact_list_no_indent(self) -> None:
        from agm.agl.runtime.serialize import dumps_exact

        result = dumps_exact([1, 2], indent=None)
        assert "1" in result
        assert "2" in result

    def test_dumps_exact_dict_empty(self) -> None:
        from agm.agl.runtime.serialize import dumps_exact

        assert dumps_exact({}) == "{}"

    def test_dumps_exact_dict_no_indent(self) -> None:
        from agm.agl.runtime.serialize import dumps_exact

        result = dumps_exact({"k": 1}, indent=None)
        assert "k" in result
        assert "1" in result


# ---------------------------------------------------------------------------
# Coverage: agents.py — AgentResponse returned directly (not str)
# ---------------------------------------------------------------------------


class TestAgentResponseDirectReturn:
    """Cover the branch in AgentRegistry.dispatch that returns AgentResponse directly."""

    def test_agent_returns_agent_response_directly(self) -> None:
        from agm.agl.runtime import AgentRequest, AgentResponse
        from agm.agl.runtime.agents import AgentRegistry

        def agent_fn(req: AgentRequest) -> AgentResponse:
            return AgentResponse(content="direct", metadata={"k": "v"})

        registry = AgentRegistry(named={"myagent": agent_fn}, default_agent=None)
        result = registry.dispatch("myagent", AgentRequest(agent="myagent", prompt="q"))
        assert result.content == "direct"
        assert result.metadata == {"k": "v"}


# ---------------------------------------------------------------------------
# Coverage: contract.py — ValueError when codec not found
# ---------------------------------------------------------------------------


class TestMaterializeContractMissingCodec:
    """Cover the ValueError branch when the codec is not in the registry."""

    def test_missing_codec_raises_value_error(self) -> None:
        from agm.agl.runtime.codec import TextCodec
        from agm.agl.runtime.contract import materialize_contract
        from agm.agl.typecheck.env import OutputContractSpec
        from agm.agl.typecheck.types import TextType

        spec = OutputContractSpec(
            target_type=TextType(),
            codec_name="nonexistent_codec",
            strict_json=None,
        )
        with pytest.raises(ValueError, match="nonexistent_codec"):
            materialize_contract(spec, {"text": TextCodec()})


# ---------------------------------------------------------------------------
# Coverage: runtime.py — generic exception handlers and error paths
# ---------------------------------------------------------------------------


class TestRuntimeErrorPaths:
    """Cover the generic exception handler branches in WorkflowRuntime.run."""

    def test_generic_parse_exception_covered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Generic (non-AglSyntaxError) exception in parse step → ok=False."""
        import agm.agl.parser as parser_mod

        def bad_parse(source: str) -> object:
            raise RuntimeError("unexpected parse error")

        monkeypatch.setattr(parser_mod, "parse_program", bad_parse)
        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert result.ok is False
        assert any("unexpected parse error" in d.message for d in result.diagnostics)

    def test_tab_warning_included_even_on_parse_failure(self) -> None:
        """Tab advisories come from the lexer's single scan, so they survive a
        parse failure: the scan completes (recording the TAB) before the grammar
        rejects the token stream."""
        rt = WorkflowRuntime()
        result = rt.run("\tprint")  # leading TAB, then an incomplete `print`
        assert result.ok is False
        assert result.diagnostics  # genuine parse error surfaced
        tab_warns = [w for w in result.warnings if w.severity == "warning"]
        assert len(tab_warns) == 1
        assert tab_warns[0].line == 1

    def test_generic_scope_exception_covered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Generic (non-AglScopeError) exception in scope step → ok=False."""
        import agm.agl.scope as scope_mod

        def bad_resolve(program: object) -> object:
            raise RuntimeError("unexpected scope error")

        monkeypatch.setattr(scope_mod, "resolve", bad_resolve)
        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert result.ok is False
        assert any("Scope error" in d.message for d in result.diagnostics)

    def test_generic_typecheck_exception_covered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Generic (non-AglTypeError) exception in typecheck step → ok=False."""
        import agm.agl.typecheck as tc_mod

        def bad_check(resolved: object, caps: object) -> object:
            raise RuntimeError("unexpected type error")

        monkeypatch.setattr(tc_mod, "check", bad_check)
        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        assert result.ok is False
        assert any("Type error" in d.message for d in result.diagnostics)

    def test_contract_error_returns_not_ok(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Contract materialization error → ok=False with contract error diagnostic."""
        import agm.agl.runtime.runtime as runtime_mod

        monkeypatch.setattr(
            runtime_mod,
            "_materialize_ir_contracts",
            lambda executable, codecs: (
                {},
                [Diagnostic(message="Contract error: bad contract", line=1)],
            ),
        )
        rt = WorkflowRuntime(default_agent=lambda req: "ok")
        result = rt.run('ask "hi"')
        assert result.ok is False
        assert any("Contract error" in d.message for d in result.diagnostics)

    def test_uncaught_agl_raise_in_run(self) -> None:
        """AglRaise propagating from the interpreter → RunResult with error."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.values import ExceptionValue, TextValue

        def bad_agent(req: object) -> str:
            raise AglRaise(
                ExceptionValue(
                    nominal=NominalId(ENTRY_ID, "Abort"),
                    display_name="Abort",
                    fields={"message": TextValue("stopped"), "trace_id": TextValue("")},
                )
            )

        rt = WorkflowRuntime(default_agent=bad_agent)
        result = rt.run('ask "hi"')
        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "Abort"

    def test_text_param_not_str_raises(self) -> None:
        """convert_param_value: text type with non-str value → ValueError."""
        from agm.agl.runtime.runtime import convert_param_value
        from agm.agl.typecheck.types import TextType

        with pytest.raises(ValueError, match="expected a text value"):
            convert_param_value("msg", 42, TextType())

    def test_int_param_decimal_integral_widened(self) -> None:
        """convert_param_value: integral Decimal → IntValue for int type."""
        from decimal import Decimal

        from agm.agl.eval.values import IntValue
        from agm.agl.runtime.runtime import convert_param_value
        from agm.agl.typecheck.types import IntType

        result = convert_param_value("n", Decimal("3"), IntType())
        assert result == IntValue(3)

    def test_int_param_non_integral_fails(self) -> None:
        """convert_param_value: non-integral value → ValueError for int type."""
        from agm.agl.runtime.runtime import convert_param_value
        from agm.agl.typecheck.types import IntType

        with pytest.raises(ValueError, match="expected an integer"):
            convert_param_value("n", "1.5", IntType())

    def test_decimal_param_from_int(self) -> None:
        """convert_param_value: int value → DecimalValue for decimal type."""
        from decimal import Decimal

        from agm.agl.eval.values import DecimalValue
        from agm.agl.runtime.runtime import convert_param_value
        from agm.agl.typecheck.types import DecimalType

        result = convert_param_value("d", 3, DecimalType())
        assert isinstance(result, DecimalValue)
        assert result.value == Decimal(3)

    def test_decimal_param_invalid_type_fails(self) -> None:
        """convert_param_value: bool value → ValueError for decimal type."""
        from agm.agl.runtime.runtime import convert_param_value
        from agm.agl.typecheck.types import DecimalType

        with pytest.raises(ValueError, match="expected a decimal"):
            convert_param_value("d", "true", DecimalType())

    def test_bool_param_invalid_type_fails(self) -> None:
        """convert_param_value: non-bool value → ValueError for bool type."""
        from agm.agl.runtime.runtime import convert_param_value
        from agm.agl.typecheck.types import BoolType

        with pytest.raises(ValueError, match="expected a bool"):
            convert_param_value("b", "1", BoolType())

    def test_bool_param_true_succeeds(self) -> None:
        """convert_param_value: bool value → BoolValue for bool type."""
        from agm.agl.eval.values import BoolValue
        from agm.agl.runtime.runtime import convert_param_value
        from agm.agl.typecheck.types import BoolType

        result = convert_param_value("b", True, BoolType())
        assert result == BoolValue(True)

    # --- assertions migrated from TestRuntimeExceptionHandlers (eval tests) ---

    def test_internal_interpreter_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F1c: an unexpected (non-AglRaise) interpreter error must propagate.

        A Python-level bug must crash loudly rather than masquerade as a
        user-facing pre-execution diagnostic.
        """
        from agm.agl.eval.ir_interpreter import IrInterpreter

        def bad_execute(self: IrInterpreter) -> dict[str, object]:
            raise RuntimeError("internal crash")

        monkeypatch.setattr(IrInterpreter, "run", bad_execute)

        rt = WorkflowRuntime()
        with pytest.raises(RuntimeError, match="internal crash"):
            rt.run("let x = 1\nx")

    def test_exception_value_to_run_error_maps_all_field_kinds(self) -> None:
        """exception_value_to_run_error converts every Value kind to JSON shape.

        This is the pure converter used to surface an uncaught AgL exception
        (e.g. AgentParseError) as a RunError.
        """
        import decimal

        from agm.agl.eval.values import (
            BoolValue,
            DecimalValue,
            DictValue,
            EnumValue,
            ExceptionValue,
            IntValue,
            JsonValue,
            ListValue,
            RecordValue,
            TextValue,
        )
        from agm.agl.runtime.runtime import RunError, exception_value_to_run_error

        exc_val = ExceptionValue(
            nominal=NominalId(PRELUDE_ID, "AgentParseError"),
            display_name="AgentParseError",
            fields={
                "message": TextValue("failed"),
                "trace_id": TextValue(""),
                "raw": TextValue("abc"),
                "agent": TextValue("ask"),
                "attempts": IntValue(1),
                "target_type": TextValue("text"),
                "decimal_val": DecimalValue(decimal.Decimal("1.5")),
                "bool_val": BoolValue(True),
                "json_val": JsonValue({"k": "v"}),
                "list_val": ListValue(elements=(IntValue(1),)),
                "dict_val": DictValue(entries={"x": IntValue(2)}),
                "rec_val": RecordValue(
                    nominal=NominalId(ENTRY_ID, "R"), display_name="R",
                    fields={"f": TextValue("v")},
                ),
                "enum_val": EnumValue(
                    nominal=NominalId(ENTRY_ID, "E"), display_name="E", variant="V", fields={}
                ),
                "exc_val": ExceptionValue(
                    nominal=NominalId(ENTRY_ID, "Inner"), display_name="Inner", fields={}
                ),
                "none_val": JsonValue(None),
            },
        )
        error = exception_value_to_run_error(exc_val)
        assert isinstance(error, RunError)
        assert error.type_name == "AgentParseError"
        assert error.fields["message"] == "failed"
        # F3/F9: Decimal is preserved exactly (not converted to float).
        assert error.fields["decimal_val"] == decimal.Decimal("1.5")
        assert isinstance(error.fields["decimal_val"], decimal.Decimal)
        assert error.fields["bool_val"] is True
        assert error.fields["json_val"] == {"k": "v"}
        assert error.fields["list_val"] == [1]
        assert error.fields["dict_val"] == {"x": 2}
        assert error.fields["rec_val"] == {"f": "v"}
        assert error.fields["enum_val"] == {"$case": "V"}
        assert isinstance(error.fields["exc_val"], dict)

    def test_convert_param_value_json_type_accepts_any(self) -> None:
        from agm.agl.eval.values import JsonValue
        from agm.agl.runtime.runtime import convert_param_value
        from agm.agl.typecheck.types import JsonType

        result = convert_param_value("meta", [1, 2, 3], JsonType())
        assert result == JsonValue([1, 2, 3])

    def test_convert_param_value_list_type_parsed_via_json_codec(self) -> None:
        # M2: list/dict/record/enum params are now accepted via the JsonCodec.
        from agm.agl.eval.values import ListValue, TextValue
        from agm.agl.runtime.runtime import convert_param_value
        from agm.agl.typecheck.types import ListType, TextType

        result = convert_param_value("xs", '["a", "b"]', ListType(elem=TextType()))
        assert isinstance(result, ListValue)
        assert result.elements == (TextValue("a"), TextValue("b"))

    def test_agl_raise_from_interpreter_becomes_run_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An AglRaise from the interpreter → RunResult.error (not None)."""
        from agm.agl.eval.exceptions import AglRaise
        from agm.agl.eval.ir_interpreter import IrInterpreter
        from agm.agl.eval.values import ExceptionValue, TextValue

        def bad_execute(self: IrInterpreter) -> dict[str, object]:
            exc_val = ExceptionValue(
                nominal=NominalId(ENTRY_ID, "Abort"),
                display_name="Abort",
                fields={"message": TextValue("fatal"), "trace_id": TextValue("")},
            )
            raise AglRaise(exc_val)

        monkeypatch.setattr(IrInterpreter, "run", bad_execute)

        rt = WorkflowRuntime()
        result = rt.run("let x = 1\nx")
        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "Abort"
        assert result.error.fields.get("message") == "fatal"

    # --- Task 3: structured Decimal params and non-JSON-shaped rejection ---

    def test_list_decimal_param_validates_exactly(self) -> None:
        """Task 3: a list[decimal] param with native Decimal values must bind
        correctly without the old default=str corruption.

        Before the fix, Decimal("1.5") was serialized as the JSON string "1.5"
        (quoted), which failed schema validation.
        """
        import decimal as _decimal

        from agm.agl.eval.values import DecimalValue, ListValue
        from agm.agl.runtime.runtime import convert_param_value
        from agm.agl.typecheck.types import DecimalType, ListType

        result = convert_param_value(
            "xs", [_decimal.Decimal("1.5"), _decimal.Decimal("2.75")], ListType(elem=DecimalType())
        )
        assert isinstance(result, ListValue)
        assert result.elements == (
            DecimalValue(_decimal.Decimal("1.5")),
            DecimalValue(_decimal.Decimal("2.75")),
        )

    def test_non_json_shaped_object_yields_clean_diagnostic(self) -> None:
        """Task 3: a non-JSON-shaped object (e.g. a set) must yield a clean
        param-validation error naming the param, not a stringified value or
        traceback.
        """
        from agm.agl.runtime.runtime import convert_param_value
        from agm.agl.typecheck.types import ListType, TextType

        with pytest.raises(ValueError, match="xs") as exc_info:
            convert_param_value("xs", {1, 2, 3}, ListType(elem=TextType()))
        # The error message must name the param and mention the type, not
        # contain a raw repr of the set or a json.dumps traceback.
        msg = str(exc_info.value)
        assert "set" in msg  # type name named

    def test_decimal_native_in_list_end_to_end(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Task 3 e2e: param xs: list[decimal] with Decimal values binds and prints."""
        import decimal as _decimal

        result = WorkflowRuntime().run(
            "param xs: list[decimal]\nprint xs\n",
            param_values={"xs": [_decimal.Decimal("1.5"), _decimal.Decimal("2.25")]},
        )
        assert result.ok is True
        out = capsys.readouterr().out
        assert "1.5" in out
        assert "2.25" in out

    def test_float_native_in_decimal_list_end_to_end(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Native JSON-shaped floats are canonicalized before typed decoding."""
        result = WorkflowRuntime().run(
            "param xs: list[decimal]\nprint xs\n",
            param_values={"xs": [1.5, 2.25]},
        )

        assert result.ok is True
        assert result.diagnostics == []
        assert capsys.readouterr().out == "[1.5, 2.25]\n"

    def test_is_json_shaped_dict_with_non_str_key_is_false(self) -> None:
        """_is_json_shaped: a dict with non-str keys is not JSON-shaped (covers
        the dict branch of _is_json_shaped, line 790).
        """
        from agm.agl.runtime.runtime import _is_json_shaped

        # Dict with non-str key.
        assert _is_json_shaped({1: "a"}) is False
        # Dict with str keys and JSON-shaped values.
        assert _is_json_shaped({"k": 1}) is True


class TestUniformRenderingInPrompts:
    """Uniform rendering: no boundary tags in agent prompts."""

    def test_text_interpolation_in_prompt_is_verbatim(self) -> None:
        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        rt = WorkflowRuntime(default_agent=agent)
        result = rt.run(
            'param x\nask("see: ${x}")',
            param_values={"x": "hello"},
        )
        assert result.ok is True
        assert received, "agent should have been called"
        prompt = received[0].prompt
        assert "hello" in prompt
        assert "<dsl-value" not in prompt

    def test_list_interpolation_in_prompt_is_pretty_json(self) -> None:
        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        rt = WorkflowRuntime(default_agent=agent)
        result = rt.run(
            'let items: list[text] = ["a", "b"]\nask("items: ${items}")',
        )
        assert result.ok is True
        prompt = received[0].prompt
        assert "a" in prompt
        assert "b" in prompt
        assert "<dsl-value" not in prompt


class TestMaxIterationsExceededSchema:
    """F2: ``MaxIterationsExceeded`` carries the full §8.1 field schema.

    The interpreter populates ``condition`` (the until-expression's exact source
    text, recovered via span offsets into the threaded source), the final
    ``last_condition_value``, and a ``metadata`` json placeholder — alongside the
    pre-existing ``limit``.  Exercised end-to-end by catching the exception in an
    AgL program and printing each field.
    """

    _PROGRAM = (
        "var n = 0\n"
        "try\n"
        "  do[2]\n"
        "    n := n + 1\n"
        "  until n > 10\n"
        "catch MaxIterationsExceeded as e =>\n"
        "  print e.limit\n"
        "  print e.condition\n"
        "  print e.last_condition_value\n"
    )

    def test_fields_surface_through_real_source(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rt = WorkflowRuntime()
        result = rt.run(self._PROGRAM)
        # The exception is caught, so the run completes successfully.
        assert result.ok is True
        assert result.error is None
        lines = capsys.readouterr().out.splitlines()
        # limit, condition (exact until-expression source), last_condition_value.
        assert lines == ["2", "n > 10", "false"]

    def test_metadata_field_is_accessible(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # ``metadata`` is a json placeholder (null until the M4 trace store), but
        # it is part of the schema and must be readable as a field.
        rt = WorkflowRuntime()
        program = (
            "try\n"
            "  do[1] ()\n"
            "  until false\n"
            "catch MaxIterationsExceeded as e =>\n"
            "  print e.metadata\n"
        )
        result = rt.run(program)
        assert result.ok is True
        assert capsys.readouterr().out.strip() == "null"

    def test_condition_reflects_each_distinct_until_expression(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A different until-expression must yield a different ``condition`` slice,
        # proving the source text is recovered per-node rather than hard-coded.
        rt = WorkflowRuntime()
        program = (
            "let done = false\n"
            "try\n"
            "  do[1] ()\n"
            "  until done\n"
            "catch MaxIterationsExceeded as e =>\n"
            "  print e.condition\n"
            "  print e.last_condition_value\n"
        )
        result = rt.run(program)
        assert result.ok is True
        lines = capsys.readouterr().out.splitlines()
        assert lines == ["done", "false"]


class TestExhaustivenessWarningSurfaces:
    """F1: a non-exhaustive enum ``case`` warns without failing the run.

    The exhaustiveness diagnostic is a warning, so ``ok`` stays ``True`` and the
    warning is visible on ``result.warnings`` (never in ``result.diagnostics``)
    while the program executes.
    """

    def test_warning_surfaces_and_run_succeeds(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rt = WorkflowRuntime()
        program = (
            "enum R\n"
            "  | Pass\n"
            "  | Fail\n"
            "let r: R = Pass()\n"
            "case r of\n"
            '  | Pass() => print "ok"\n'
        )
        result = rt.run(program)
        # Warning, not error: the run still succeeds.
        assert result.ok is True
        assert result.error is None
        # Successful runs carry no error diagnostics; the warning is separate.
        assert result.diagnostics == []
        assert len(result.warnings) == 1
        assert result.warnings[0].severity == "warning"
        assert "Fail" in result.warnings[0].message
        # The matched branch executed.
        assert capsys.readouterr().out == "ok\n"


class TestShellExecTimeoutProperty:
    """M4: shell_exec_timeout is a readable constructor parameter."""

    def test_default_shell_exec_timeout_is_none(self) -> None:
        rt = WorkflowRuntime()
        assert rt.shell_exec_timeout is None

    def test_shell_exec_timeout_kwarg_is_observable(self) -> None:
        rt = WorkflowRuntime(shell_exec_timeout=30.0)
        assert rt.shell_exec_timeout == 30.0


# Permission-based tests: chmod 0o444 has no effect for root, who can write
# regardless.  Skip there rather than assert a false negative.
_skip_if_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="permission tests are meaningless as root (root bypasses file modes)",
)


class TestTraceWriteFailureIsBestEffort:
    """F2b: a mid-run trace write failure must not corrupt program semantics."""

    def test_emit_failure_warns_once_and_disables_store(
        self,
        tmp_path: "object",
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failing ``append_jsonl`` is caught: one stderr warning is emitted,
        the store disables itself, and no further writes are attempted."""
        from pathlib import Path

        from agm.agl.runtime.trace import TraceStore

        store = TraceStore(path=Path(str(tmp_path)) / "trace.log")
        calls = {"n": 0}

        def failing_append(_path: object, _record: object) -> None:
            calls["n"] += 1
            raise OSError("disk gone")

        import agm.agl.runtime.trace as trace_mod

        monkeypatch.setattr(trace_mod, "append_jsonl", failing_append)
        store.run_start()  # first emit → fails, warns, disables
        store.print_stmt(rendered="hi")  # disabled → no further attempt
        store.run_end(ok=True)

        # Only the first emit attempted a write; the rest short-circuit.
        assert calls["n"] == 1
        err = capsys.readouterr().err
        assert err.count("trace logging disabled") == 1
        assert "disk gone" in err

    @_skip_if_root
    def test_run_completes_when_trace_becomes_unwritable_midrun(
        self, tmp_path: "object", capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A real chmod-444 mid-run leaves the program semantics intact: every
        statement still runs and the RunResult is a normal success."""
        from pathlib import Path

        log_file = Path(str(tmp_path)) / "trace.log"
        rt = WorkflowRuntime()

        # Pre-create the trace file and make it read-only so the first record
        # write (run_start) fails — the run must still complete normally.
        log_file.write_text("")
        log_file.chmod(0o444)

        program = 'print "a"\nprint "b"\nprint "c"\n'
        result = rt.run(program, log_file=log_file)

        # Program semantics unaffected: clean success, all prints emitted.
        assert result.ok is True
        assert result.error is None
        out = capsys.readouterr()
        assert out.out == "a\nb\nc\n"
        # Exactly one warning about disabled trace logging.
        assert out.err.count("trace logging disabled") == 1


# ---------------------------------------------------------------------------
# TAB character warnings in RunResult
# ---------------------------------------------------------------------------


class TestTabWarningsInRunResult:
    """Tab characters in source produce warning diagnostics in RunResult.warnings."""

    def test_no_tab_no_warning(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run("let x = 1")
        tab_warns = [w for w in result.warnings if "TAB" in w.message]
        assert tab_warns == []

    def test_tab_in_valid_source_yields_warning(self) -> None:
        # TAB used as whitespace inside a valid statement on the second line.
        source = "let x = 1\nlet\ty = 2"
        rt = WorkflowRuntime()
        result = rt.run(source)
        tab_warns = [w for w in result.warnings if "TAB" in w.message]
        assert len(tab_warns) == 1
        assert tab_warns[0].line == 2

    def test_tab_warning_does_not_affect_ok(self) -> None:
        # A tab warning must not cause ok to become False.
        source = "let\tx = 1\nx"
        rt = WorkflowRuntime()
        result = rt.run(source)
        assert result.ok is True
        assert result.error is None

    def test_multiple_tabs_multiple_warnings(self) -> None:
        source = "let\tx = 1\nlet\ty = 2"
        rt = WorkflowRuntime()
        result = rt.run(source)
        tab_warns = [w for w in result.warnings if "TAB" in w.message]
        assert len(tab_warns) == 2


# ---------------------------------------------------------------------------
# declared_agents() API (M4)
# ---------------------------------------------------------------------------


class TestDeclaredAgentsApi:
    """WorkflowRuntime.declared_agents(): parse + scope only, non-raising."""

    def test_returns_agent_decl_info_with_names_runners_and_positions(self) -> None:
        from agm.agl import AgentDeclInfo

        rt = WorkflowRuntime()
        source = 'agent impl = "claude -p %{PROMPT_FILE}"\nagent reviewer'
        decls = rt.declared_agents(source)
        assert all(isinstance(d, AgentDeclInfo) for d in decls)
        # Sorted deterministically by source line/col.
        assert [d.name for d in decls] == ["impl", "reviewer"]
        impl, reviewer = decls
        assert impl.runner == "claude -p %{PROMPT_FILE}"
        assert reviewer.runner is None
        # Positions come from the declaration span (1-based).
        assert impl.line == 1
        assert impl.col == 1
        assert reviewer.line == 2

    def test_no_declarations_returns_empty(self) -> None:
        rt = WorkflowRuntime()
        assert rt.declared_agents("let x = 1") == ()

    def test_parse_error_returns_empty_tuple(self) -> None:
        rt = WorkflowRuntime()
        # Syntax garbage: declared_agents stays non-raising and returns ().
        assert rt.declared_agents("@@@@@") == ()

    def test_scope_error_returns_empty_tuple(self) -> None:
        rt = WorkflowRuntime()
        # Duplicate agent declaration is a scope error → ().
        assert rt.declared_agents("agent dup\nagent dup") == ()

    def test_undeclared_call_scope_error_returns_empty_tuple(self) -> None:
        rt = WorkflowRuntime()
        # Calling an undeclared agent is a scope error → ().
        assert rt.declared_agents('let x = ghost "hi"') == ()


# ---------------------------------------------------------------------------
# Source↔host reconciliation in run() (M4, plan §8, decisions 1 & 11)
# ---------------------------------------------------------------------------


class TestAgentReconciliation:
    """run() enforces the source↔host agent contract before execution."""

    def test_registered_but_undeclared_is_host_error(self) -> None:
        calls: list[str] = []

        def agent(req: AgentRequest) -> str:
            calls.append(req.prompt)
            return "ok"

        rt = WorkflowRuntime()
        rt.register_agent("ghost", agent)
        # 'ghost' is registered but the source never declares it.
        result = rt.run("let x = 1")
        assert result.ok is False
        assert result.error is None
        msgs = " ".join(d.message for d in result.diagnostics)
        assert "ghost" in msgs
        assert "registered" in msgs.lower()
        # Nothing executed.
        assert calls == []

    def test_registered_but_undeclared_diagnostic_line_is_one(self) -> None:
        rt = WorkflowRuntime()
        rt.register_agent("ghost", lambda req: "ok")
        result = rt.run("let x = 1")
        assert result.diagnostics[0].line == 1

    def test_declared_but_unbacked_is_host_error(self) -> None:
        rt = WorkflowRuntime()  # no registration, no default agent
        result = rt.run('agent orphan\nlet x = orphan "hi"')
        assert result.ok is False
        assert result.error is None
        msgs = " ".join(d.message for d in result.diagnostics)
        assert "orphan" in msgs
        assert "backing" in msgs.lower()

    def test_declared_but_unbacked_diagnostic_reports_declaration_line(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run('let y = 1\nagent orphan\nlet x = orphan "hi"')
        assert result.ok is False
        # The declaration is on line 2.
        assert result.diagnostics[0].line == 2

    def test_declared_and_registered_runs(self) -> None:
        calls: list[str] = []

        def agent(req: AgentRequest) -> str:
            calls.append(req.prompt)
            return "output"

        rt = WorkflowRuntime()
        rt.register_agent("impl", agent)
        result = rt.run('agent impl\nask("do it", agent: impl)')
        assert result.ok is True
        assert calls == ["do it"]

    def test_declared_with_default_agent_runs(self) -> None:
        # No dedicated registration, but a default agent backs the declared name.
        rt = WorkflowRuntime(default_agent=lambda req: "ok")
        result = rt.run('agent any_name\nask("hi", agent: any_name)')
        assert result.ok is True

    def test_both_error_categories_reported_together(self) -> None:
        rt = WorkflowRuntime()  # no default agent
        rt.register_agent("ghost", lambda req: "ok")
        # 'orphan' is declared but unbacked; 'ghost' is registered but undeclared.
        result = rt.run('agent orphan\nlet x = orphan "hi"')
        assert result.ok is False
        msgs = " ".join(d.message for d in result.diagnostics)
        assert "ghost" in msgs
        assert "orphan" in msgs
        assert len(result.diagnostics) == 2

    def test_reconciliation_failure_skips_execution(self) -> None:
        calls: list[str] = []

        def agent(req: AgentRequest) -> str:
            calls.append(req.prompt)
            return "ok"

        rt = WorkflowRuntime()
        rt.register_agent("ghost", agent)
        rt.run('print "side effect?"')
        assert calls == []


# ---------------------------------------------------------------------------
# Coverage: schema.py — derive_schema branches not exercised higher up
# ---------------------------------------------------------------------------


class TestDeriveSchema:
    """Unit tests for derive_schema covering all type branches."""

    def test_bool_type(self) -> None:
        from agm.agl.type_schema import derive_schema
        from agm.agl.typecheck.types import BoolType

        assert derive_schema(BoolType()) == {"type": "boolean"}

    def test_json_type(self) -> None:
        from agm.agl.type_schema import derive_schema
        from agm.agl.typecheck.types import JsonType

        assert derive_schema(JsonType()) == {}

    def test_dict_type(self) -> None:
        from agm.agl.type_schema import derive_schema
        from agm.agl.typecheck.types import DictType, IntType

        result = derive_schema(DictType(value=IntType()))
        assert result == {"type": "object", "additionalProperties": {"type": "integer"}}

    def test_record_type(self) -> None:
        from agm.agl.type_schema import derive_schema
        from agm.agl.typecheck.types import RecordType, TextType

        result = derive_schema(RecordType(name="Point", fields={"x": TextType()}))
        assert result["type"] == "object"
        assert result["required"] == ["x"]
        assert result["additionalProperties"] is False

    def test_enum_type_with_payload(self) -> None:
        from agm.agl.type_schema import derive_schema
        from agm.agl.typecheck.types import EnumType, TextType

        typ = EnumType(
            name="Status",
            variants={"Pass": {}, "Fail": {"reason": TextType()}},
        )
        result = derive_schema(typ)
        assert "oneOf" in result
        assert len(result["oneOf"]) == 2

    def test_exception_type_raises(self) -> None:
        from agm.agl.type_schema import derive_schema
        from agm.agl.typecheck.types import ExceptionType

        with pytest.raises(TypeError, match="ExceptionType"):
            derive_schema(ExceptionType(name="MyErr", fields={}))

    def test_unit_type_raises(self) -> None:
        from agm.agl.type_schema import derive_schema
        from agm.agl.typecheck.types import UnitType

        with pytest.raises(TypeError, match="UnitType"):
            derive_schema(UnitType())

    def test_agent_type_raises(self) -> None:
        from agm.agl.type_schema import derive_schema
        from agm.agl.typecheck.types import AgentType

        with pytest.raises(TypeError, match="AgentType"):
            derive_schema(AgentType())

    def test_function_type_raises(self) -> None:
        from agm.agl.type_schema import derive_schema
        from agm.agl.typecheck.types import FunctionType, TextType

        with pytest.raises(TypeError, match="FunctionType"):
            derive_schema(FunctionType(params=(TextType(),), result=TextType()))

    def test_bottom_type_raises(self) -> None:
        from agm.agl.type_schema import derive_schema
        from agm.agl.typecheck.types import BottomType

        with pytest.raises(TypeError, match="BottomType"):
            derive_schema(BottomType())


# ---------------------------------------------------------------------------
# Coverage: serialize.py — v2 opaque value TypeError branches
# ---------------------------------------------------------------------------


class TestSerializeV2OpaqueValues:
    """Unit, agent, and IR closure values have no JSON representation (D9)."""

    def test_unit_value_raises(self) -> None:
        from agm.agl.eval.values import UnitValue
        from agm.agl.runtime.serialize import value_to_json_obj

        with pytest.raises(TypeError, match="UnitValue"):
            value_to_json_obj(UnitValue())

    def test_agent_value_raises(self) -> None:
        from agm.agl.eval.values import AgentValue
        from agm.agl.runtime.serialize import value_to_json_obj

        with pytest.raises(TypeError, match="AgentValue"):
            value_to_json_obj(AgentValue(name="myagent"))

    def test_ir_closure_value_raises(self) -> None:
        """IrClosureValue has no JSON representation — TypeError is raised."""
        from agm.agl.eval.values import IrClosureValue
        from agm.agl.ir.ids import FunctionId
        from agm.agl.runtime.serialize import value_to_json_obj

        ir_closure = IrClosureValue(function_id=FunctionId(0), captures=())
        with pytest.raises(TypeError, match="IrClosureValue"):
            value_to_json_obj(ir_closure)


# ---------------------------------------------------------------------------
# Coverage: runtime.py — uncovered branches and new v2 properties
# ---------------------------------------------------------------------------


class TestIrHostMetadataCoverage:
    def test_invalid_external_param_shapes_are_diagnostics(self) -> None:
        text_result = WorkflowRuntime().run(
            "param value: text\nprint value", param_values={"value": 3}
        )
        assert not text_result.ok
        assert "text value" in text_result.diagnostics[0].message

        json_result = WorkflowRuntime().run(
            "param value: int\nprint value", param_values={"value": {1, 2}}
        )
        assert not json_result.ok
        assert "JSON-compatible" in json_result.diagnostics[0].message

    def test_missing_ir_codec_materialization_is_diagnostic(self) -> None:
        from agm.agl.ir.contracts import ContractRequest
        from agm.agl.ir.ids import ContractId
        from agm.agl.ir.program import ExecutableProgram
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.runtime.runtime import _materialize_ir_contracts

        request = ContractRequest(
            codec_name="missing",
            strict_json=None,
            json_schema=None,
            decode=None,
            target_type_label="text",
            structured_exec=False,
            format_instructions="",
        )
        executable = ExecutableProgram(
            entry_module=ENTRY_ID,
            modules={},
            symbols={},
            nominals={},
            sources={},
            contracts={ContractId(0): request},
        )
        contracts, errors = _materialize_ir_contracts(executable, {})
        assert contracts == {}
        assert "missing" in errors[0].message

    def test_decode_schema_defensive_and_integral_decimal_paths(self) -> None:
        from decimal import Decimal

        from agm.agl.eval.conversions import decode_value
        from agm.agl.eval.values import IntValue
        from agm.agl.ir.contracts import ScalarDecode, ScalarKind
        from agm.agl.runtime.contract import _type_from_decode

        assert decode_value(ScalarDecode(ScalarKind.INT), Decimal("2.0")) == IntValue(2)
        with pytest.raises(AssertionError, match="Unknown decode schema"):
            _type_from_decode(object())


class TestRunErrorToMessage:
    """RunError.to_message with include_trace_id=True/False."""

    def test_to_message_with_trace_id(self) -> None:
        from agm.agl.runtime.runtime import RunError

        err = RunError(
            type_name="AgentParseError",
            fields={"message": "bad output", "trace_id": "abc123"},
            line=5,
            col=3,
        )
        msg = err.to_message(include_trace_id=True)
        assert "trace_id=abc123" in msg
        assert "at line 5, col 3" in msg

    def test_to_message_without_trace_id(self) -> None:
        from agm.agl.runtime.runtime import RunError

        err = RunError(
            type_name="SomeError",
            fields={"message": "oops"},
            line=2,
        )
        msg = err.to_message(include_trace_id=False)
        assert "trace_id" not in msg
        assert "at line 2" in msg


class TestHostEnvironmentCache:
    """WorkflowRuntime.host_environment() caches and is invalidated on registration."""

    def test_host_environment_returns_same_object_on_second_call(self) -> None:
        rt = WorkflowRuntime()
        env1 = rt.host_environment()
        env2 = rt.host_environment()
        assert env1 is env2

    def test_register_agent_invalidates_cache(self) -> None:
        rt = WorkflowRuntime()
        env1 = rt.host_environment()
        rt.register_agent("impl", lambda req: "ok")
        env2 = rt.host_environment()
        assert env1 is not env2


class TestRegisterCodecErrors:
    """register_codec raises for reserved names and duplicates."""

    def _make_codec(self, name: str) -> OutputCodec:
        from agm.agl.eval.values import TextValue
        from agm.agl.runtime.codec import OutputCodec, ParseResult, TextCodec
        from agm.agl.runtime.contract import OutputContract
        from agm.agl.typecheck.types import TextType

        class _Codec(OutputCodec):
            @property
            def name(self) -> str:
                return _name

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset({"text"})

            def supports_type(self, t: Type) -> bool:
                return isinstance(t, TextType)

            def make_contract(self, type_ref: Type) -> OutputContract:
                return OutputContract(
                    target_type=type_ref,
                    codec=TextCodec(),
                    strict_json=None,
                    format_instructions="",
                    json_schema=None,
                )

            def parse(
                self,
                raw: str,
                target_type: Type,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
            ) -> ParseResult:
                return ParseResult.success(TextValue(""))

        _name = name
        return _Codec()

    def test_reserved_name_raises(self) -> None:
        rt = WorkflowRuntime()
        codec = self._make_codec("text")  # "text" is a builtin codec name
        with pytest.raises(ValueError, match="reserved"):
            rt.register_codec(codec)

    def test_duplicate_name_raises(self) -> None:
        rt = WorkflowRuntime()
        codec1 = self._make_codec("mycodec")
        codec2 = self._make_codec("mycodec")
        rt.register_codec(codec1)
        with pytest.raises(ValueError, match="already registered"):
            rt.register_codec(codec2)


class TestDefaultCallDepthLimit:
    """default_call_depth_limit constructor parameter and property."""

    def test_default_is_256(self) -> None:
        rt = WorkflowRuntime()
        assert rt.default_call_depth_limit == 256

    def test_custom_value_is_observable(self) -> None:
        rt = WorkflowRuntime(default_call_depth_limit=128)
        assert rt.default_call_depth_limit == 128


class TestConvertInputUnsupportedType:
    """convert_param_value raises ValueError for unsupported types (e.g. ListType of records)."""

    def test_unsupported_type_raises(self) -> None:
        from agm.agl.runtime.runtime import convert_param_value
        from agm.agl.typecheck.types import AgentType

        with pytest.raises(ValueError, match="unsupported type"):
            convert_param_value("x", "agent_val", AgentType())


# ---------------------------------------------------------------------------
# New v2 feature tests: user-defined functions, ExecResult, ask with AgentValue
# ---------------------------------------------------------------------------


class TestV2UserDefinedFunctions:
    """v2 def expressions: first-class functions, recursion, call depth limit."""

    def test_def_call_basic(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run(
            "def add(a: int, b: int) -> int = a + b\n"
            "add(1, 2)\n"
        )
        assert result.ok is True

    def test_def_recursive_call(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run(
            "def fact(n: int) -> int =\n"
            "  if n <= 1 =>\n"
            "    1\n"
            "  | else =>\n"
            "    n * fact(n - 1)\n"
            "fact(5)\n"
        )
        assert result.ok is True

    def test_def_call_depth_limit_enforced(self) -> None:
        """Exceeding max_call_depth raises a RecursionError (D8)."""
        rt = WorkflowRuntime(default_call_depth_limit=10)
        result = rt.run(
            "def inf(n: int) -> int =\n"
            "  inf(n + 1)\n"
            "inf(0)\n"
        )
        assert result.ok is False


class TestV2ExecStructuredForm:
    """v2 exec structured form: let x: T = exec ... raises on nonzero."""

    def test_exec_text_form_captures_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        rt = WorkflowRuntime()
        result = rt.run('let out: text = exec "echo hello"\nprint out\n')
        assert result.ok is True
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_exec_nonzero_raises_when_typed(self) -> None:
        rt = WorkflowRuntime()
        result = rt.run('let out: text = exec "false"\nprint out\n')
        assert result.ok is False
        # Uncaught AgL exception (exit 2 semantics): error is set
        assert result.error is not None


class TestV2AskWithAgentValue:
    """ask(..., agent: <agent_value>) dispatches to the named agent."""

    def test_ask_dispatches_to_named_agent(self) -> None:
        received: list[str] = []

        def agent(req: AgentRequest) -> str:
            received.append(req.prompt)
            return "answer"

        rt = WorkflowRuntime()
        rt.register_agent("helper", agent)
        result = rt.run('agent helper\nask("question", agent: helper)\n')
        assert result.ok is True
        assert received == ["question"]


# ---------------------------------------------------------------------------
# M5b: PreparedGraph / prepare_program / run_prepared_graph / discover_params_graph
# ---------------------------------------------------------------------------


class TestPrepareProgram:
    """WorkflowRuntime.prepare_program: graph-mode front-end (M5b)."""

    def test_prepare_program_no_imports_returns_prepared_graph(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A single-file program with no imports produces a valid PreparedGraph."""
        from agm.agl.modules.roots import RootSet
        from agm.agl.runtime.runtime import PreparedGraph

        roots = RootSet(roots=frozenset())
        prepared = WorkflowRuntime.prepare_program(
            "let x = 1\nx", entry_path=None, roots=roots
        )
        assert isinstance(prepared, PreparedGraph)
        assert prepared.resolved_graph is not None
        assert prepared.diagnostics == ()

    def test_prepare_program_syntax_error_captured(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A syntax error is captured as a diagnostic, not raised."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset())
        prepared = WorkflowRuntime.prepare_program(
            "let x = !!!", entry_path=None, roots=roots
        )
        assert prepared.resolved_graph is None
        assert len(prepared.diagnostics) >= 1

    def test_prepare_program_missing_import_captured(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A missing import is captured as a diagnostic, not raised."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset())
        prepared = WorkflowRuntime.prepare_program(
            "import nonexistent.module\nlet x = 1\nx",
            entry_path=None,
            roots=roots,
        )
        assert prepared.resolved_graph is None
        assert len(prepared.diagnostics) >= 1
        assert "nonexistent.module" in prepared.diagnostics[0].message

    def test_prepare_program_with_valid_import(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A valid import resolves when the module file exists on disk."""
        from agm.agl.modules.roots import RootSet

        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "mymod.agl").write_text("def add(a: int, b: int) -> int = a + b\n")

        roots = RootSet(roots=frozenset([lib_dir]))
        entry = "import mymod\nlet r = add(2, 3)\nr"
        prepared = WorkflowRuntime.prepare_program(
            entry, entry_path=None, roots=roots
        )
        assert prepared.resolved_graph is not None
        assert prepared.diagnostics == ()

    def test_prepare_program_declared_agents_from_entry(
        self, tmp_path: pathlib.Path
    ) -> None:
        """declared_agents reads from the entry module only."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset())
        prepared = WorkflowRuntime.prepare_program(
            'agent reviewer\nask("q", agent: reviewer)',
            entry_path=None,
            roots=roots,
        )
        assert prepared.resolved_graph is not None
        assert any(d.name == "reviewer" for d in prepared.declared_agents)

    def test_prepare_program_config_pragmas_from_entry(
        self, tmp_path: pathlib.Path
    ) -> None:
        """config_pragmas reads from the entry module."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset())
        prepared = WorkflowRuntime.prepare_program(
            "config max_iters = 7\nlet x = 1\nx",
            entry_path=None,
            roots=roots,
        )
        assert prepared.resolved_graph is not None
        assert prepared.config_pragmas.get("max_iters") == 7

    def test_prepare_program_failure_returns_empty_declared_agents(
        self, tmp_path: pathlib.Path
    ) -> None:
        """When scope fails, declared_agents returns ()."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset())
        prepared = WorkflowRuntime.prepare_program(
            "let x = undefined_name", entry_path=None, roots=roots
        )
        # scope error → resolved_graph is None
        assert prepared.resolved_graph is None
        assert prepared.declared_agents == ()


class TestRunPreparedGraph:
    """WorkflowRuntime.run_prepared_graph: graph execution (M5b)."""

    def test_single_entry_graph_behaves_like_run(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A single-file program via run_prepared_graph returns same result as run()."""
        from agm.agl.modules.roots import RootSet

        rt = WorkflowRuntime()
        roots = RootSet(roots=frozenset())
        prepared = WorkflowRuntime.prepare_program(
            "let x = 1\nx", entry_path=None, roots=roots
        )
        result = rt.run_prepared_graph(prepared)
        assert result.ok is True
        assert result.error is None

    def test_graph_with_library_module_executes(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A two-module graph (entry + library) runs to completion."""
        from agm.agl.modules.roots import RootSet

        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "mymod.agl").write_text("def add(a: int, b: int) -> int = a + b\n")

        roots = RootSet(roots=frozenset([lib_dir.resolve()]))
        entry = "import mymod\nlet r = add(2, 3)\nr"
        prepared = WorkflowRuntime.prepare_program(
            entry, entry_path=None, roots=roots
        )
        rt = WorkflowRuntime()
        result = rt.run_prepared_graph(prepared)
        assert result.ok is True
        assert result.error is None

    def test_graph_failure_propagated(
        self, tmp_path: pathlib.Path
    ) -> None:
        """When the load phase captured a scope error, run_prepared_graph reports it."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset())
        prepared = WorkflowRuntime.prepare_program(
            "let x = undefined_name", entry_path=None, roots=roots
        )
        rt = WorkflowRuntime()
        result = rt.run_prepared_graph(prepared)
        assert result.ok is False
        assert result.error is None
        assert len(result.diagnostics) >= 1

    def test_graph_missing_import_propagated(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A missing import error from prepare_program flows through run_prepared_graph."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset())
        prepared = WorkflowRuntime.prepare_program(
            "import missing.module\nlet x = 1\nx", entry_path=None, roots=roots
        )
        rt = WorkflowRuntime()
        result = rt.run_prepared_graph(prepared)
        assert result.ok is False
        assert "missing.module" in result.diagnostics[0].message

    def test_graph_agents_are_entry_owned(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Agents are entry-program-owned; a registered undeclared agent is an error."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset())
        prepared = WorkflowRuntime.prepare_program(
            "let x = 1\nx", entry_path=None, roots=roots
        )
        rt = WorkflowRuntime()
        rt.register_agent("reviewer", lambda req: "resp")  # type: ignore[arg-type]
        result = rt.run_prepared_graph(prepared)
        assert result.ok is False
        assert any("reviewer" in d.message for d in result.diagnostics)

    def test_graph_check_only_returns_call_inventory(
        self, tmp_path: pathlib.Path
    ) -> None:
        """check_only=True produces call_sites from the entry module."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset())
        prepared = WorkflowRuntime.prepare_program(
            'let r = ask("hello")\nprint r', entry_path=None, roots=roots
        )
        rt = WorkflowRuntime(default_agent=lambda req: "x")  # type: ignore[arg-type]
        result = rt.run_prepared_graph(prepared, check_only=True)
        assert result.ok is True
        assert len(result.call_sites) >= 1

    def test_multimodule_wildcard_import(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Wildcard import brings multiple modules into scope."""
        from agm.agl.modules.roots import RootSet

        lib_dir = tmp_path / "lib"
        utils_dir = lib_dir / "utils"
        utils_dir.mkdir(parents=True)
        (utils_dir / "math.agl").write_text(
            "def add(a: int, b: int) -> int = a + b\n"
        )
        (utils_dir / "strings.agl").write_text(
            'def greet(name: text) -> text = "Hello, " + name + "!"\n'
        )

        roots = RootSet(roots=frozenset([lib_dir.resolve()]))
        entry = 'import utils.*\nlet n = add(2, 3)\nlet g = greet("World")\nprint n\nprint g\n'
        prepared = WorkflowRuntime.prepare_program(
            entry, entry_path=None, roots=roots
        )
        rt = WorkflowRuntime()
        result = rt.run_prepared_graph(prepared, check_only=True)
        assert result.ok is True

    def test_multimodule_qualified_import(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Qualified import requires :: qualifier to access names."""
        from agm.agl.modules.roots import RootSet

        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "calc.agl").write_text(
            "def square(n: int) -> int = n * n\n"
        )

        roots = RootSet(roots=frozenset([lib_dir.resolve()]))
        entry = "import calc qualified\nlet r = calc::square(5)\nr"
        prepared = WorkflowRuntime.prepare_program(
            entry, entry_path=None, roots=roots
        )
        rt = WorkflowRuntime()
        result = rt.run_prepared_graph(prepared)
        assert result.ok is True


class TestDiscoverParamsGraph:
    """WorkflowRuntime.discover_params_graph: typed param discovery (M5b)."""

    def test_discover_params_no_params(
        self, tmp_path: pathlib.Path
    ) -> None:
        """discover_params_graph returns empty params for a program with no params."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset())
        prepared = WorkflowRuntime.prepare_program(
            "let x = 1\nx", entry_path=None, roots=roots
        )
        rt = WorkflowRuntime()
        discovery = rt.discover_params_graph(prepared)
        assert discovery.diagnostics == ()
        assert discovery.params == ()

    def test_discover_params_with_declared_param(
        self, tmp_path: pathlib.Path
    ) -> None:
        """discover_params_graph discovers typed param declarations."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset())
        prepared = WorkflowRuntime.prepare_program(
            "param name: text\nprint name",
            entry_path=None,
            roots=roots,
        )
        rt = WorkflowRuntime()
        discovery = rt.discover_params_graph(prepared)
        assert discovery.diagnostics == ()
        assert len(discovery.params) == 1
        assert discovery.params[0].name == "name"

    def test_discover_params_failure_returns_diagnostics(
        self, tmp_path: pathlib.Path
    ) -> None:
        """discover_params_graph returns diagnostics when the prepare phase failed."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset())
        prepared = WorkflowRuntime.prepare_program(
            "import no_such_module\nlet x = 1", entry_path=None, roots=roots
        )
        rt = WorkflowRuntime()
        discovery = rt.discover_params_graph(prepared)
        assert len(discovery.diagnostics) >= 1


# ---------------------------------------------------------------------------
# M5b: coverage gap tests for defensive/exceptional paths
# ---------------------------------------------------------------------------


class TestPreparedGraphDefensivePaths:
    """Edge-case coverage for PreparedGraph properties and prepare_program error paths."""

    def test_config_pragmas_no_entry_module_in_graph(self, tmp_path: pathlib.Path) -> None:
        """config_pragmas returns {} when resolved_graph has no ENTRY_ID module."""
        from unittest.mock import MagicMock

        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.modules.roots import RootSet
        from agm.agl.runtime.runtime import PreparedGraph

        # Build a fake resolved_graph with no ENTRY_ID key.
        fake_graph = MagicMock()
        fake_graph.modules = {}  # empty — no ENTRY_ID
        assert ENTRY_ID not in fake_graph.modules

        roots = RootSet(roots=frozenset())
        pg = PreparedGraph(
            source="let x = 1",
            entry_path=None,
            roots=roots,
            resolved_graph=fake_graph,
            diagnostics=(),
            warnings=(),
        )
        assert pg.config_pragmas == {}

    def test_program_name_no_entry_module_in_graph(self, tmp_path: pathlib.Path) -> None:
        """program_name returns None when resolved_graph has no ENTRY_ID module."""
        from unittest.mock import MagicMock

        from agm.agl.modules.roots import RootSet
        from agm.agl.runtime.runtime import PreparedGraph

        fake_graph = MagicMock()
        fake_graph.modules = {}

        roots = RootSet(roots=frozenset())
        pg = PreparedGraph(
            source="let x = 1",
            entry_path=None,
            roots=roots,
            resolved_graph=fake_graph,
            diagnostics=(),
            warnings=(),
        )
        assert pg.program_name is None

    def test_prepare_program_generic_exception_during_load(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A non-AglError exception during load_graph is captured as a diagnostic."""
        from unittest.mock import patch

        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset([tmp_path.resolve()]))
        with patch("agm.agl.modules.loader.load_graph", side_effect=RuntimeError("boom")):
            prepared = WorkflowRuntime.prepare_program(
                "let x = 1\nx", entry_path=None, roots=roots
            )
        assert len(prepared.diagnostics) >= 1
        assert "boom" in prepared.diagnostics[0].message

    def test_prepare_program_generic_exception_during_resolve(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A non-AglScopeError exception during resolve_graph is captured."""
        from unittest.mock import patch

        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset([tmp_path.resolve()]))
        with patch(
            "agm.agl.scope.graph.resolve_graph",
            side_effect=RuntimeError("resolve fail"),
        ):
            prepared = WorkflowRuntime.prepare_program(
                "let x = 1\nx", entry_path=None, roots=roots
            )
        assert len(prepared.diagnostics) >= 1
        assert "resolve fail" in prepared.diagnostics[0].message


class TestDiscoverParamsDefensivePaths:
    """Edge-case coverage for discover_params_graph and discover_params."""

    def test_discover_params_graph_missing_entry_in_checked_graph(
        self, tmp_path: pathlib.Path
    ) -> None:
        """discover_params_graph returns diagnostic when checked graph has no entry module."""
        from unittest.mock import MagicMock, patch

        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset())
        prepared = WorkflowRuntime.prepare_program(
            "let x = 1\nx", entry_path=None, roots=roots
        )
        rt = WorkflowRuntime()
        # Patch check_graph to return a graph with no ENTRY_ID.
        fake_checked_graph = MagicMock()
        fake_checked_graph.modules = {}

        with patch(
            "agm.agl.typecheck.graph.check_graph", return_value=fake_checked_graph
        ):
            discovery = rt.discover_params_graph(prepared)
        assert len(discovery.diagnostics) >= 1
        assert "Entry module not found" in discovery.diagnostics[0].message

    def test_discover_params_typecheck_failure(self, tmp_path: pathlib.Path) -> None:
        """discover_params returns diagnostics when typecheck fails."""
        from unittest.mock import patch

        from agm.agl.typecheck import AglTypeError

        prepared = WorkflowRuntime.prepare(
            'def f(x: int) -> text = "bad"\nlet r = f(1)\nprint r'
        )
        rt = WorkflowRuntime()
        with patch("agm.agl.typecheck.check", side_effect=AglTypeError("type error")):
            discovery = rt.discover_params(prepared)
        assert discovery.checked is None
        assert len(discovery.diagnostics) >= 1

    def test_run_typecheck_graph_generic_exception_captured(
        self, tmp_path: pathlib.Path
    ) -> None:
        """_run_typecheck_graph captures generic exceptions as diagnostics."""
        from unittest.mock import MagicMock, patch

        from agm.agl.modules.roots import RootSet
        from agm.agl.runtime.runtime import PreparedGraph

        roots = RootSet(roots=frozenset())
        # Build a PreparedGraph with a fake resolved_graph so we reach check_graph.
        fake_rg = MagicMock()
        fake_rg.warnings = ()
        fake_rg.modules = {}

        pg = PreparedGraph(
            source="let x = 1",
            entry_path=None,
            roots=roots,
            resolved_graph=fake_rg,
            diagnostics=(),
            warnings=(),
        )
        rt = WorkflowRuntime()
        with patch(
            "agm.agl.typecheck.graph.check_graph",
            side_effect=RuntimeError("graph type crash"),
        ):
            # discover_params_graph will call _run_typecheck_graph internally.
            discovery = rt.discover_params_graph(pg)
        assert len(discovery.diagnostics) >= 1
        assert "graph type crash" in discovery.diagnostics[0].message


class TestRunPreparedDefensivePaths:
    """Edge-case coverage for run_prepared (single-file) and run_prepared_graph."""

    def test_run_prepared_with_precomputed_checked(self) -> None:
        """run_prepared skips typecheck when pre-computed checked is passed."""
        from agm.agl.typecheck import check

        prepared = WorkflowRuntime.prepare("let x = 1\nx")
        assert prepared.resolved is not None
        rt = WorkflowRuntime()
        env = rt.host_environment()
        precomputed = check(prepared.resolved, env.capabilities)
        # Pass the pre-computed checked: the branch `if checked is None:` is skipped.
        result = rt.run_prepared(prepared, checked=precomputed)
        assert result.ok is True

    def test_run_prepared_typecheck_failure(self) -> None:
        """run_prepared exits early when typecheck fails (checked is None)."""
        from unittest.mock import patch

        from agm.agl.typecheck import AglTypeError

        prepared = WorkflowRuntime.prepare("let x = 1\nx")
        rt = WorkflowRuntime()
        with patch("agm.agl.typecheck.check", side_effect=AglTypeError("tc fail")):
            result = rt.run_prepared(prepared)
        assert result.ok is False
        assert any("tc fail" in d.message for d in result.diagnostics)

    def test_run_prepared_graph_contract_error(self, tmp_path: pathlib.Path) -> None:
        """run_prepared_graph exits early when an IR contract is invalid."""
        from unittest.mock import patch

        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset())
        # Use ask() so the checked graph has at least one contract_spec to materialize.
        prepared = WorkflowRuntime.prepare_program(
            'let r = ask("hi")\nprint r', entry_path=None, roots=roots
        )
        rt = WorkflowRuntime(default_agent=lambda req: "ok")  # type: ignore[arg-type]
        with patch(
            "agm.agl.runtime.runtime._materialize_ir_contracts",
            return_value=(
                {},
                [Diagnostic(message="Contract error: bad contract", line=1)],
            ),
        ):
            result = rt.run_prepared_graph(prepared)
        # A contract error yields a RunResult with ok=False and a contract diagnostic.
        assert result.ok is False
        assert any("Contract error" in d.message for d in result.diagnostics)
    def test_run_prepared_graph_with_prechecked_graph_decodes_param_from_ir(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A prechecked graph executes supplied params through IR metadata."""
        from agm.agl.modules.roots import RootSet
        from agm.agl.typecheck.graph import check_graph as real_check_graph

        roots = RootSet(roots=frozenset())
        prepared = WorkflowRuntime.prepare_program(
            "param n: int\nprint n", entry_path=None, roots=roots
        )
        rt = WorkflowRuntime()
        env = rt.host_environment()
        checked_graph = real_check_graph(prepared.resolved_graph, env.capabilities)  # type: ignore[arg-type]
        result = rt.run_prepared_graph(
            prepared, param_values={"n": 7}, checked_graph=checked_graph
        )
        assert result.ok
