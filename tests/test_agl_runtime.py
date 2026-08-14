"""Tests for PipelineDriver shell behaviors and runtime additions.

Covers:
- PipelineDriver constructor with default kwargs
- register_agent: duplicate rejection, reserved-name rejection (ask, exec)
- run: full pipeline now active; valid programs succeed; static errors fail
- Diagnostic has .message (str) and .line (int)
- RunResult has .ok (bool), .diagnostics (list), .error (None for pre-exec failures)
- AglError and SourceSpan
- Token constants
- Runtime additions: agent registration/fallback, capability derivation, param validation,
  text-codec behavior, AgentCallError, empty-response valid case
"""

from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING

import pytest

from agm.agl import AglError, PipelineDriver, SourceSpan
from agm.agl.diagnostics import Diagnostic, format_diagnostic, format_diagnostic_location
from agm.agl.ir.ids import NominalId
from agm.agl.pipeline import RunResult
from agm.agl.runtime import AgentRequest
from agm.agl.runtime.contract import OutputContract
from agm.agl.semantics.types import Type
from agm.agl.semantics.values import TextValue
from agm.agl.typecheck import AglTypeError
from agm.commands import exec_program as exec_engine
from tests._agl_helpers import prepare_inline_command, run_inline_command, type_table_for

if TYPE_CHECKING:
    from agm.agl.runtime.codec import OutputCodec

_STDLIB_ROOT = pathlib.Path(__file__).resolve().parents[1] / "stdlib"


class TestOperatorProgramsRunEndToEnd:
    """Operator declaration, application, and `is`-test dispatch through a full run."""

    def test_operator_name_bindings_run_end_to_end(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = run_inline_command(
            PipelineDriver(),
            "let >> = 0\n"
            "def |>[T](x: T, f: T -> T) -> T = f x\n"
            "print(>>)\n"
            "print((|>)(1, fn(y: int) -> int => y))\n",
        )

        assert result.ok is True
        assert result.diagnostics == []
        assert capsys.readouterr().out == "0\n1\n"

    def test_user_defined_left_infix_operator_run_end_to_end(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = run_inline_command(
            PipelineDriver(),
            "infixl |> at prio > + 1\n"
            "def |>(x: int, y: int) -> int = x * 10 + y\n"
            "print(1 + 2 |> 3 > 20)\n",
        )

        assert result.ok is True
        assert result.diagnostics == []
        assert capsys.readouterr().out == "true\n"

    def test_is_test_uses_scrutinee_owner_for_shared_variant_name(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = run_inline_command(
            PipelineDriver(),
            "enum Left\n  | Same\n"
            "enum Right\n  | Same\n"
            "let value = Left::Same()\n"
            "print(value is Same)\n",
        )

        assert result.ok is True
        assert capsys.readouterr().out == "true\n"

    def test_user_defined_right_infix_operator_run_end_to_end(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = run_inline_command(
            PipelineDriver(),
            "infixr << at 40\n"
            'def <<(x: text, y: text) -> text = "(" + x + y + ")"\n'
            'print("a" << "b" << "c")\n',
        )

        assert result.ok is True
        assert result.diagnostics == []
        assert capsys.readouterr().out == "(a(bc))\n"


class TestRegisterAgent:
    def test_register_agent_accepted(self) -> None:
        prompts: list[str] = []

        def my_agent(request: AgentRequest) -> str:
            prompts.append(request.prompt)
            return "response"

        rt = PipelineDriver(agent_dispatcher=my_agent)
        result = run_inline_command(
            rt,
            'let my_agent = AgentCommand("my_agent")\n'
            'let answer = ask("meaningful prompt", agent = my_agent)\nprint answer',
        )

        assert result.ok
        assert prompts == ["meaningful prompt"]


class TestRunBehavior:
    """run() behavior: valid programs run, static errors fail cleanly."""

    def test_run_returns_run_result(self) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, "let x = 1")
        assert isinstance(result, RunResult)

    def test_valid_program_ok(self) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, "let x = 1\nx")
        assert result.ok is True

    def test_run_uses_the_standard_library_program(self) -> None:
        result = run_inline_command(PipelineDriver(), "Option::Some(value = 1)")
        assert result.ok is True

    def test_static_error_not_ok(self) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, "let x = undefined_name")
        assert result.ok is False
        assert result.error is None
        assert len(result.diagnostics) >= 1

    def test_static_error_diagnostic_has_message(self) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, "let x = undefined_name")
        diag = result.diagnostics[0]
        assert isinstance(diag.message, str)
        assert diag.message

    def test_static_error_diagnostic_has_line(self) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, "let x = undefined_name")
        diag = result.diagnostics[0]
        assert isinstance(diag.line, int)
        assert diag.line >= 1

    def test_run_result_error_none_for_static_failure(self) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, "let x = undefined_name")
        # pre-execution failure: error is None (no AgL exception was raised)
        assert result.error is None

    def test_run_with_params(self) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, "param k\nprint k", param_values={"k": "value"})
        assert isinstance(result, RunResult)
        assert result.ok is True

    def test_run_with_empty_params(self) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, "let x = 1\nx", param_values={})
        assert isinstance(result, RunResult)
        assert result.ok is True

    def test_run_parse_error_not_ok(self) -> None:
        rt = PipelineDriver()
        # Invalid syntax
        result = run_inline_command(rt, "@@@@@")
        assert result.ok is False
        assert result.error is None


class TestFallbackAgent:
    """Default-agent backing behavior for capability checking."""

    def test_ask_without_a_dispatcher_raises_at_runtime(self) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, 'let x = ask "hi"')
        assert result.ok is False
        assert result.error is not None

    def test_with_default_agent_ask_call_succeeds(self) -> None:
        rt = PipelineDriver(agent_dispatcher=lambda req: "ok")
        result = run_inline_command(rt, 'let x = ask "hi"\nx')
        assert result.ok is True

    def test_named_agent_registered_accepted(self) -> None:
        rt = PipelineDriver(agent_dispatcher=lambda req: "output")
        result = run_inline_command(
            rt, 'let impl = AgentCommand("impl")\nask("do it", agent = impl)'
        )
        assert result.ok is True

    def test_undeclared_named_agent_is_static_error(self) -> None:
        rt = PipelineDriver()
        # An undeclared named agent is a static scope binding error: it is
        # rejected before execution regardless of host backing.
        result = run_inline_command(rt, 'let x = mysterious_agent "hi"')
        assert result.ok is False
        assert result.error is None

    def test_default_agent_dispatches_an_agent_value(self) -> None:
        rt = PipelineDriver(agent_dispatcher=lambda req: "ok")
        result = run_inline_command(
            rt,
            (
                'let any_agent_name = AgentCommand("any-agent-name")\n'
                'ask("hi", agent = any_agent_name)'
            ),
        )
        assert result.ok is True


class TestInputValidationRuntime:
    """Param validation before execution."""

    def test_missing_param_fails_not_ok(self) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, "param spec\nprint spec", param_values={})
        assert result.ok is False
        assert result.error is None  # host error, not AgL exception

    def test_missing_param_mentions_name(self) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, "param spec\nprint spec", param_values={})
        msgs = " ".join(d.message for d in result.diagnostics)
        assert "spec" in msgs.lower()

    def test_undeclared_extra_is_ignored(self) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, "param a\nprint a", param_values={"a": "ok", "b": "extra"})
        assert result.ok is True
        assert result.diagnostics == []

    def test_text_param_verbatim(self) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, "param msg\nprint msg", param_values={"msg": "hello world"})
        assert result.ok is True

    def test_no_agent_called_on_param_failure(self) -> None:
        calls: list[str] = []

        def agent(req: AgentRequest) -> str:
            calls.append(req.prompt)
            return "ok"

        rt = PipelineDriver(agent_dispatcher=agent)
        run_inline_command(rt, 'param x\nask("Hi")', param_values={})
        assert calls == []

    def test_int_param_json_parsed(self, capsys: pytest.CaptureFixture[str]) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, "param n: int\nprint n", param_values={"n": 5})
        assert result.ok
        out = capsys.readouterr().out
        assert "5" in out

    def test_invalid_typed_param_fails(self) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, "param n: int\nprint n", param_values={"n": "five"})
        assert result.ok is False
        assert result.error is None

    def test_missing_param_reports_declaration_line(self) -> None:
        """the missing-param diagnostic carries the declaration's line."""
        rt = PipelineDriver()
        # ``param spec`` is on line 3; the diagnostic must report line 3, not 1.
        src = "let a = 1\nlet b = 2\nparam spec\nprint spec"
        result = run_inline_command(rt, src, param_values={})
        assert result.ok is False
        missing = [d for d in result.diagnostics if "spec" in d.message.lower()]
        assert missing, result.diagnostics
        assert missing[0].line == 3

    def test_invalid_typed_param_reports_declaration_line(self) -> None:
        """parity: the type-invalid diagnostic already reports the line."""
        rt = PipelineDriver()
        src = "let a = 1\nlet b = 2\nparam n: int\nprint n"
        result = run_inline_command(rt, src, param_values={"n": "five"})
        assert result.ok is False
        bad = [d for d in result.diagnostics if "n" in d.message.lower()]
        assert bad, result.diagnostics
        assert bad[0].line == 3


class TestEmptyResponse:
    """Exit 0 with empty stdout is a valid empty response."""

    def test_empty_string_response_is_valid_text(self) -> None:
        rt = PipelineDriver(agent_dispatcher=lambda req: "")
        result = run_inline_command(rt, 'let x = ask "Say nothing."\nx')
        assert result.ok is True
        from agm.agl.semantics.values import TextValue

        assert result.bindings["x"] == TextValue("")


class TestAgentRequest:
    """AgentRequest contract: .prompt and .agent fields."""

    def test_request_prompt_is_rendered_template(self) -> None:
        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        rt = PipelineDriver(agent_dispatcher=agent)
        run_inline_command(rt, 'ask "Hello world"')
        assert received[0].prompt == "Hello world"

    def test_request_agent_value_for_default(self) -> None:
        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        rt = PipelineDriver(agent_dispatcher=agent)
        run_inline_command(rt, 'ask "Hi"')
        assert received[0].agent.variant == "AgentClaude"

    def test_request_agent_value_for_named(self) -> None:
        received: list[AgentRequest] = []

        def reviewer(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        rt = PipelineDriver(agent_dispatcher=reviewer)
        run_inline_command(
            rt, 'let reviewer = AgentCommand("reviewer")\nask("Review this", agent = reviewer)'
        )
        assert received[0].agent.variant == "AgentCommand"
        assert received[0].agent.fields["command"] == TextValue("reviewer")


class TestUncaughtAgentCallErrorSpan:
    """an uncaught AgentCallError carries the agent-call site's location.

    ``AgentRegistry.dispatch`` raises ``AglRaise`` without a span; the
    interpreter must attach the agent-call node's span so the exit-2 error
    reports ``at line N``.
    """

    def _failing_runtime(self) -> PipelineDriver:
        from agm.agl.runtime.agents import AgentCallHostError

        def failing_agent(req: AgentRequest) -> str:
            raise AgentCallHostError(
                cause="spawn_failure",
                exit_code=None,
                stderr_tail="boom",
                elapsed=0.0,
            )

        return PipelineDriver(agent_dispatcher=failing_agent)

    def test_dispatch_preserves_existing_span(self) -> None:
        """A span the raise site already supplied is never overwritten."""
        from agm.agl.semantics.exceptions import AglRaise
        from agm.agl.semantics.values import ExceptionValue, TextValue

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
                nominal=NominalId(1),
                display_name="CustomError",
                fields={"message": TextValue("boom")},
            )
            raise AglRaise(exc_val, span=existing)

        rt = PipelineDriver(agent_dispatcher=agent)
        result = run_inline_command(rt, 'let a = 1\nask("hi")')
        assert result.ok is False
        assert result.error is not None
        # The agent's own span (line 99) is kept, not replaced by the call site.
        assert result.error.line == 99

    def test_uncaught_agent_call_error_reports_call_line(self) -> None:
        rt = self._failing_runtime()
        # The ``ask`` call is on line 2.
        result = run_inline_command(rt, 'let a = 1\nask("hi")')
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
        from agm.agl.runtime.agents import AgentCallHostError
        from agm.cli_support.args import ExecArgs
        from agm.commands.exec import run as exec_run

        agl_file = tmp_path / "prog.agl"
        from tests._agl_helpers import write_file_program

        write_file_program(agl_file, 'let a = 1\nask("hi")\n')

        def failing_agent(req: object) -> str:
            raise AgentCallHostError(
                cause="spawn_failure", exit_code=None, stderr_tail="boom", elapsed=0.0
            )

        monkeypatch.setattr(exec_engine, "value_driven_agent_factory", lambda **_: failing_agent)
        args = ExecArgs(
            file=str(agl_file),
            param_tokens=[],
            strict_json=None,
            max_iters=None,
            no_log=True,
            log_file=None,
        )
        with pytest.raises(SystemExit) as exc_info:
            exec_run(args)
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert "at line 3" in err


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
        rt = PipelineDriver()
        result = run_inline_command(rt, "let x = 1\nx")
        assert hasattr(result, "bindings")
        assert isinstance(result.bindings, dict)


class TestSourceSpan:
    def test_source_span_attributes(self) -> None:
        span = SourceSpan(
            start_line=1,
            start_col=1,
            end_line=1,
            end_col=11,
            start_offset=0,
            end_offset=10,
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
            start_line=5,
            start_col=3,
            end_line=5,
            end_col=12,
            start_offset=40,
            end_offset=49,
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
            start_line=2,
            start_col=1,
            end_line=2,
            end_col=5,
            start_offset=10,
            end_offset=14,
        )
        err = AglError("test", span=span)
        assert err.span is span

    def test_agl_error_no_span_is_none(self) -> None:
        err = AglError("test")
        assert err.span is None

    def test_related_notes_are_retained_by_type_errors(self) -> None:
        primary = SourceSpan(1, 1, 1, 2, 0, 1)
        related = SourceSpan(2, 3, 2, 4, 4, 5)

        diagnostic = AglTypeError(
            "conflicting constraint",
            span=primary,
            related=(("previous constraint", related),),
        ).to_diagnostic()

        assert diagnostic.related[0].message == "previous constraint"
        assert diagnostic.related[0].line == 2
        assert diagnostic.related[0].column == 3


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


class TestResetExternRegistry:
    """``PipelineDriver.reset_extern_registry`` — the ``ReplSession.reset()`` seam."""

    def test_noop_before_the_host_environment_is_ever_assembled(self) -> None:
        # Nothing cached yet: resetting must not crash, and a later
        # ``host_environment()`` call still works normally afterward.
        rt = PipelineDriver()
        rt.reset_extern_registry()
        assert rt.host_environment().extern_registry is not None

    def test_replaces_the_cached_registry_with_a_fresh_one(self) -> None:
        rt = PipelineDriver()
        before = rt.host_environment().extern_registry
        rt.reset_extern_registry()
        after = rt.host_environment().extern_registry
        assert after is not before


class TestNoDefaultAgent:
    """an ``ask`` call needs a default (or fallback) agent."""

    def test_ask_without_a_dispatcher_is_a_runtime_error(self) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, 'ask "hi"')
        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "AgentCallError"

    def test_ask_with_default_agent_runs(self) -> None:
        def agent(request: object) -> str:
            return "answer"

        rt = PipelineDriver(agent_dispatcher=agent)
        result = run_inline_command(rt, 'ask "hi"')
        assert result.ok is True
        assert result.error is None

    def test_later_solved_ask_uses_its_concrete_contract_at_runtime(self) -> None:
        requests: list[AgentRequest] = []

        def agent(request: AgentRequest) -> str:
            requests.append(request)
            return "7"

        result = run_inline_command(
            PipelineDriver(agent_dispatcher=agent),
            "def select[T](first: T, second: T) -> T = first\n"
            'let value = select(ask("number"), 1)\n'
            "value",
        )
        assert result.ok is True
        assert len(requests) == 1
        contract = requests[0].output_contract
        assert isinstance(contract, OutputContract)
        assert contract.codec.name == "json"
        assert contract.json_schema == {"type": "integer"}


class TestDryRunCheckOnly:
    """``check_only=True`` runs the static pipeline but executes nothing."""

    def test_check_only_printing_program_produces_no_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, 'print "hello"', check_only=True)
        assert result.ok is True
        assert result.bindings == {}
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_check_only_static_error_still_fails(self) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, "let x = undefined_name", check_only=True)
        assert result.ok is False

    def test_check_only_never_invokes_agent(self) -> None:
        calls: list[object] = []

        def agent(request: object) -> str:
            calls.append(request)
            return "should not be called"

        rt = PipelineDriver(agent_dispatcher=agent)
        result = run_inline_command(rt, 'let x = ask "hi"\nx', check_only=True)
        assert result.ok is True
        # The agent must never be invoked during a dry run.
        assert calls == []

    def test_check_only_unit_ask_reports_no_codec(self) -> None:
        rt = PipelineDriver(agent_dispatcher=lambda request: "ignored")
        result = run_inline_command(rt, 'let result: unit = ask "hi"\nresult', check_only=True)
        assert result.ok is True
        assert len(result.call_sites) == 1
        site = result.call_sites[0]
        assert site.target_type == "unit"
        assert site.codec_name == "none"
        assert site.has_schema is False

    def test_check_only_param_validation_still_runs(self) -> None:
        rt = PipelineDriver()
        # Missing declared param is caught even under check_only.
        result = run_inline_command(rt, "param msg\nprint msg", param_values={}, check_only=True)
        assert result.ok is False
        assert any("msg" in d.message for d in result.diagnostics)


class TestDecimalSerialization:
    """decimals print/round-trip exactly; never via binary float."""

    def test_json_param_with_decimal_prints_exactly(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rt = PipelineDriver()
        result = run_inline_command(
            rt, "param data: json\nprint data", param_values={"data": '{"a": 1.5}'}
        )
        assert result.ok is True
        captured = capsys.readouterr()
        assert "1.5" in captured.out
        # No binary-float artifacts (e.g. 1.5000000000000002).
        assert "1.5000" not in captured.out

    def test_decimal_value_prints_exact_text(self, capsys: pytest.CaptureFixture[str]) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, "let x = 0.1\nprint x")
        assert result.ok is True
        captured = capsys.readouterr()
        assert captured.out.strip() == "0.1"

    def test_run_error_preserves_decimal_exactness(self) -> None:
        import decimal

        from agm.agl.pipeline import exception_value_to_run_error
        from agm.agl.semantics.values import DecimalValue, ExceptionValue, TextValue

        exc = ExceptionValue(
            nominal=NominalId(1),
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
    """typecheck warnings survive param-validation failure paths."""

    def test_warning_and_missing_param_both_visible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Inject a checker warning through the
        # CheckedModule the runtime threads from ``check``.  This exercises the
        # real failure path (missing param) while a warning is present.
        from dataclasses import replace

        import agm.agl.typecheck.program as tc_mod
        from agm.agl.diagnostics import Diagnostic

        real_check = tc_mod.check_program
        warning = Diagnostic(message="a checker warning", line=1, severity="warning")

        def check_with_warning(resolved: object, caps: object) -> object:
            checked = real_check(resolved, caps)
            return replace(checked, warnings=(*checked.warnings, warning))

        monkeypatch.setattr(tc_mod, "check_program", check_with_warning)

        rt = PipelineDriver()
        result = run_inline_command(rt, "param msg\nprint msg", param_values={})
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
    """dispatch resolves named agents, ask, and the default fallback."""


class TestAgentRequestFieldOrder:
    """The request dataclass keeps its documented positional field order."""


class TestParamBindingInvariant:
    """The runtime relies on the checker recording every param's binding type."""

    def test_missing_binding_type_is_internal_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agm.agl.typecheck.env import TypeEnvironment

        # Force the checker invariant to be violated: no recorded binding type.
        monkeypatch.setattr(TypeEnvironment, "get_binding_type", lambda self, node_id: None)

        rt = PipelineDriver()
        with pytest.raises(AssertionError, match="binding type"):
            run_inline_command(rt, "param msg\nprint msg", param_values={"msg": "hi"})


# ---------------------------------------------------------------------------
# Capabilities built from registrations
# ---------------------------------------------------------------------------


class TestCapabilitiesBuiltFromRegistrations:
    """PipelineDriver.run builds HostCapabilities from codec/renderer registries."""

    def test_default_runtime_has_text_and_json_codecs(self) -> None:
        """Built-in text + json codecs are always present."""
        from agm.agl.runtime.codec import JsonCodec, TextCodec

        rt = PipelineDriver(agent_dispatcher=lambda req: "ok")
        # A json-typed call passes typecheck → json codec is registered.
        tc, jc = TextCodec(), JsonCodec()
        assert tc.name == "text"
        assert jc.name == "json"
        result = run_inline_command(rt, 'ask "hi"')
        assert result.ok is True

    def test_register_codec_before_run_extends_capabilities(self) -> None:
        """A custom codec registered before run() makes its kinds available to typecheck."""
        from agm.agl.runtime.codec import ParseResult, TextCodec
        from agm.agl.runtime.contract import OutputContract
        from agm.agl.semantics.type_table import TypeTable
        from agm.agl.semantics.types import TextType
        from agm.agl.semantics.values import TextValue as TV

        class FooCodec:
            @property
            def name(self) -> str:
                return "foo"

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset({"text"})

            def supports_type(self, t: Type) -> bool:
                return isinstance(t, TextType)

            def make_contract(
                self, type_ref: Type, type_table: TypeTable | None = None
            ) -> OutputContract:
                return OutputContract(
                    target_type_label=repr(type_ref),
                    codec=TextCodec(),
                    strict_json=None,
                    format_instructions="",
                    json_schema=None,
                )

            def parse(
                self,
                raw: str,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
                decode: object | None = None,
            ) -> ParseResult:
                return ParseResult.success(TV(raw))

        rt = PipelineDriver()
        rt.register_codec(FooCodec())
        # The program just needs to run without capability errors.
        result = run_inline_command(rt, "let x = 1\nx")
        assert result.ok is True

    def test_custom_codec_named_none_materializes_its_contract(self) -> None:
        from agm.agl.runtime.codec import TextCodec
        from agm.agl.semantics.type_table import TypeTable
        from agm.agl.semantics.types import TextType
        from agm.agl.semantics.values import TextValue

        class NoneCodec(TextCodec):
            @property
            def name(self) -> str:
                return "none"

            def make_contract(
                self, type_ref: Type, type_table: TypeTable | None = None
            ) -> OutputContract:
                assert isinstance(type_ref, TextType)
                return OutputContract(
                    target_type_label=repr(type_ref),
                    codec=self,
                    strict_json=None,
                    format_instructions="Return a none-codec response.",
                    json_schema=None,
                )

        requests: list[AgentRequest] = []

        def agent(request: AgentRequest) -> str:
            requests.append(request)
            return "decoded"

        rt = PipelineDriver(agent_dispatcher=agent)
        rt.register_codec(NoneCodec())
        result = run_inline_command(rt, 'let answer: text = ask("prompt", format = "none")\nanswer')

        assert result.ok
        assert result.bindings["answer"] == TextValue("decoded")
        assert len(requests) == 1
        assert requests[0].output_contract is not None
        assert requests[0].output_contract.format_instructions == "Return a none-codec response."

    def test_as_renderer_syntax_is_parse_error(self) -> None:
        """``%{x as name}`` is a syntax error (renderer syntax removed)."""
        rt = PipelineDriver(agent_dispatcher=lambda req: "ok")
        result = run_inline_command(
            rt, 'param x\nlet y = ask "see %{x as fancy}"', param_values={"x": "hi"}
        )
        assert result.ok is False


# ---------------------------------------------------------------------------
# Coverage: render.py — render_value / _scalar_text
# ---------------------------------------------------------------------------


class TestRenderValue:
    """Unit tests for the AgL-native render_value function.

    Nominal values carry their fields in declaration order already (the
    interpreter normalizes them at construction), so these tests build the
    values in the order they expect rendered and the renderer walks
    ``value.fields`` verbatim — no type lookup is involved.
    """

    # ------------------------------------------------------------------
    # text: verbatim by default, quoted for display when requested
    # ------------------------------------------------------------------

    def test_text_verbatim_render_value(self) -> None:
        """render_value: top-level text is verbatim, no quotes."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import TextValue

        assert render_value(TextValue("hello world")) == "hello world"

    def test_text_quoted_when_requested(self) -> None:
        """render_value: top-level text can be wrapped in double quotes."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import TextValue

        assert render_value(TextValue("hello"), quote_strings=True) == '"hello"'

    def test_text_quote_escapes_special_chars(self) -> None:
        """render_value: JSON escape set applies to quoted top-level text."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import TextValue

        assert render_value(TextValue('a"b'), quote_strings=True) == '"a\\"b"'
        assert render_value(TextValue("a\\b"), quote_strings=True) == '"a\\\\b"'
        assert render_value(TextValue("a\nb"), quote_strings=True) == '"a\\nb"'
        assert render_value(TextValue("a\tb"), quote_strings=True) == '"a\\tb"'

    def test_text_quote_escapes_control_chars_as_unicode(self) -> None:
        """render_value: control chars below 0x20 render as \\uXXXX when quoted."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import TextValue

        assert render_value(TextValue("a\x00b"), quote_strings=True) == '"a\\u0000b"'

    # ------------------------------------------------------------------
    # Scalars: int / decimal / bool (unchanged at any depth)
    # ------------------------------------------------------------------

    def test_int_value_is_plain_text(self) -> None:
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import IntValue

        assert render_value(IntValue(42)) == "42"

    def test_decimal_value_is_plain_text(self) -> None:
        from decimal import Decimal

        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import DecimalValue

        assert render_value(DecimalValue(Decimal("1.5"))) == "1.5"

    def test_bool_value_is_plain_text(self) -> None:
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import BoolValue

        assert render_value(BoolValue(True)) == "true"
        assert render_value(BoolValue(False)) == "false"

    # ------------------------------------------------------------------
    # _scalar_text unit tests (unchanged helper — keep working)
    # ------------------------------------------------------------------

    def test_scalar_text_int(self) -> None:
        """_scalar_text(IntValue) renders as plain decimal digits."""
        from agm.agl.runtime.render import _scalar_text
        from agm.agl.semantics.values import IntValue

        assert _scalar_text(IntValue(42)) == "42"

    def test_scalar_text_decimal(self) -> None:
        """_scalar_text(DecimalValue) drops trailing zeros, no sci notation."""
        from decimal import Decimal

        from agm.agl.runtime.render import _scalar_text
        from agm.agl.semantics.values import DecimalValue

        assert _scalar_text(DecimalValue(Decimal("1.50"))) == "1.5"
        assert _scalar_text(DecimalValue(Decimal("100"))) == "100"

    def test_scalar_text_bool(self) -> None:
        """_scalar_text(BoolValue) renders as 'true'/'false'."""
        from agm.agl.runtime.render import _scalar_text
        from agm.agl.semantics.values import BoolValue

        assert _scalar_text(BoolValue(True)) == "true"
        assert _scalar_text(BoolValue(False)) == "false"

    # ------------------------------------------------------------------
    # unit / agent / closure (non-data, unchanged)
    # ------------------------------------------------------------------

    def test_unit_value_renders_as_unit_literal(self) -> None:
        """Unit value renders as ``()``."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import VOID_VALUE, UnitValue

        assert render_value(UnitValue()) == "()"
        assert render_value(VOID_VALUE) == "void"

    def test_closure_renders_as_function_surface_form(self) -> None:
        """Closure renders as ``<function: (A, B) -> T>``."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import IrClosureValue

        rt = PipelineDriver()
        result = run_inline_command(rt, "let f = fn(x: int, y: int) -> int => x + y\nf\n")
        assert result.ok is True
        closure = result.bindings["f"]
        assert isinstance(closure, IrClosureValue)
        assert render_value(closure) == "<function: (int, int) -> int>"

    def test_closure_single_arg_omits_arg_parentheses(self) -> None:
        """A single-parameter closure renders as ``<function: A -> T>``."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import IrClosureValue

        rt = PipelineDriver()
        result = run_inline_command(rt, "let f = fn(x: int) -> int => x + 1\nf\n")
        assert result.ok is True
        closure = result.bindings["f"]
        assert isinstance(closure, IrClosureValue)
        assert render_value(closure) == "<function: int -> int>"

    def test_closure_function_arg_parenthesized(self) -> None:
        """A function-typed parameter stays parenthesized for unambiguous rendering."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import IrClosureValue

        rt = PipelineDriver()
        result = run_inline_command(rt, "let f = fn(g: int -> int) -> int => g(1)\nf\n")
        assert result.ok is True
        closure = result.bindings["f"]
        assert isinstance(closure, IrClosureValue)
        assert render_value(closure) == "<function: (int -> int) -> int>"

    def test_closure_multi_param_function_arg_parenthesized(self) -> None:
        """A single multi-parameter function argument keeps both required paren pairs."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import IrClosureValue

        rt = PipelineDriver()
        result = run_inline_command(rt, "let f = fn(g: (int, int) -> int) -> int => g(1, 2)\nf\n")
        assert result.ok is True
        closure = result.bindings["f"]
        assert isinstance(closure, IrClosureValue)
        assert render_value(closure) == "<function: ((int, int) -> int) -> int>"

    def test_closure_container_arg_with_nested_function_omits_outer_parentheses(self) -> None:
        """Only top-level arrows force parentheses around a single parameter label."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import IrClosureValue

        rt = PipelineDriver()
        result = run_inline_command(rt, "let f = fn(gs: array[int -> int]) -> int => 0\nf\n")
        assert result.ok is True
        closure = result.bindings["f"]
        assert isinstance(closure, IrClosureValue)
        assert render_value(closure) == "<function: array[int -> int] -> int>"

    def test_closure_zero_arity_renders_correctly(self) -> None:
        """A zero-parameter closure renders as ``<function: () -> T>``."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import IrClosureValue

        rt = PipelineDriver()
        result = run_inline_command(rt, "let thunk = fn() -> int => 42\nthunk\n")
        assert result.ok is True
        closure = result.bindings["thunk"]
        assert isinstance(closure, IrClosureValue)
        assert render_value(closure) == "<function: () -> int>"

    def test_closure_without_param_metadata_renders_unknown_arg_types(self) -> None:
        """A hand-built closure without type labels uses ``?`` placeholders."""
        from agm.agl.ir.ids import FunctionId
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import IrClosureValue

        closure = IrClosureValue(
            function_id=FunctionId(0),
            captures=(),
            arity=2,
            result_label="int",
        )

        assert render_value(closure) == "<function: (?, ?) -> int>"

    # ------------------------------------------------------------------
    # array: single-line AgL form
    # ------------------------------------------------------------------

    def test_array_single_line(self) -> None:
        """Arrays render as single-line ``[e1, e2, ...]``."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import ArrayValue, IntValue

        v = ArrayValue(elements=[IntValue(1), IntValue(2)])
        assert render_value(v) == "[1, 2]"

    def test_array_empty(self) -> None:
        """Empty array renders as ``[]``."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import ArrayValue

        assert render_value(ArrayValue(elements=[])) == "[]"

    def test_array_nested_text_is_quoted(self) -> None:
        """Text inside an array is quoted."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import ArrayValue, TextValue

        v = ArrayValue(elements=[TextValue("tests"), TextValue("coverage")])
        assert render_value(v) == '["tests", "coverage"]'

    # ------------------------------------------------------------------
    # dict: always-quoted keys, single-line AgL form
    # ------------------------------------------------------------------

    def test_dict_single_line_quoted_keys(self) -> None:
        """Dict renders single-line with always-quoted keys."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import DictValue, IntValue

        v = DictValue({"origin": IntValue(1)})
        assert render_value(v) == '{"origin": 1}'

    def test_dict_key_with_space_is_quoted(self) -> None:
        """Dict keys with spaces (or any content) are always quoted."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import DictValue, IntValue

        v = DictValue({"two words": IntValue(1)})
        assert render_value(v) == '{"two words": 1}'

    def test_dict_empty(self) -> None:
        """Empty dict renders as ``{}``."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import DictValue

        assert render_value(DictValue({})) == "{}"

    def test_dict_multiple_entries(self) -> None:
        """Multiple dict entries, in insertion order, keys quoted."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import DictValue, IntValue

        v = DictValue({"origin": IntValue(1), "two words": IntValue(2)})
        assert render_value(v) == '{"origin": 1, "two words": 2}'

    # ------------------------------------------------------------------
    # record: AgL form, declaration order
    # ------------------------------------------------------------------

    def test_record_renders_fields_in_stored_order(self) -> None:
        """Record renders fields in the order they are stored (declaration order)."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import IntValue, RecordValue, TextValue

        v = RecordValue(
            nominal=NominalId(1),
            display_name="Issue",
            fields={"title": TextValue("Missing tests"), "severity": IntValue(3)},
        )
        assert render_value(v) == 'Issue(title = "Missing tests", severity = 3)'

    def test_record_empty(self) -> None:
        """A record with no fields renders as ``TypeName()``."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import RecordValue

        v = RecordValue(nominal=NominalId(1), display_name="Empty", fields={})
        assert render_value(v) == "Empty()"

    def test_record_nested_record(self) -> None:
        """Nested records render inline."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import BoolValue, RecordValue, TextValue

        author = RecordValue(
            nominal=NominalId(1),
            display_name="Author",
            fields={"name": TextValue("Ada"), "active": BoolValue(True)},
        )
        issue = RecordValue(
            nominal=NominalId(2),
            display_name="Issue",
            fields={"title": TextValue("Missing tests"), "author": author},
        )
        out = render_value(issue)
        assert out == 'Issue(title = "Missing tests", author = Author(name = "Ada", active = true))'

    def test_record_with_array_field(self) -> None:
        """Record with an array field renders correctly (array inline)."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import ArrayValue, IntValue, RecordValue, TextValue

        v = RecordValue(
            nominal=NominalId(1),
            display_name="Issue",
            fields={
                "title": TextValue("Missing tests"),
                "severity": IntValue(3),
                "tags": ArrayValue(elements=[TextValue("tests"), TextValue("coverage")]),
            },
        )
        out = render_value(v)
        assert out == 'Issue(title = "Missing tests", severity = 3, tags = ["tests", "coverage"])'

    # ------------------------------------------------------------------
    # enum: qualified form, nullary variant
    # ------------------------------------------------------------------

    def test_enum_with_payload(self) -> None:
        "Enum with payload renders as ``TypeName::Variant(field = value)``."
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import EnumValue, IntValue

        v = EnumValue(
            nominal=NominalId(1),
            display_name="Outcome",
            variant="Partial",
            fields={"left": IntValue(2)},
        )
        assert render_value(v) == "Outcome::Partial(left = 2)"

    def test_enum_nullary_variant(self) -> None:
        "Nullary enum variant renders as ``TypeName::Variant`` (no parens)."
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import EnumValue

        v = EnumValue(
            nominal=NominalId(1),
            display_name="Outcome",
            variant="Done",
            fields={},
        )
        assert render_value(v) == "Outcome::Done"

    def test_enum_multi_field_payload(self) -> None:
        """Enum with multiple payload fields renders them in stored order."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import EnumValue, IntValue

        v = EnumValue(
            nominal=NominalId(1),
            display_name="E",
            variant="V",
            fields={"a": IntValue(1), "b": IntValue(2), "c": IntValue(3)},
        )
        assert render_value(v) == "E::V(a = 1, b = 2, c = 3)"

    # ------------------------------------------------------------------
    # exception: record-style with declared fields
    # ------------------------------------------------------------------

    def test_exception_renders_record_style(self) -> None:
        """Exception renders like a record with all fields in declaration order."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import ExceptionValue, TextValue

        v = ExceptionValue(
            nominal=NominalId(1),
            display_name="CastError",
            fields={
                "message": TextValue('cannot parse "x" as int'),
                "source_type": TextValue("text"),
                "target_type": TextValue("int"),
                "raw": TextValue("x"),
            },
        )
        out = render_value(v)
        expected = (
            'CastError(message = "cannot parse \\"x\\" as int", '
            'source_type = "text", target_type = "int", raw = "x")'
        )
        assert out == expected

    def test_exception_abort_renders_with_message(self) -> None:
        """Abort exception renders its sole message field."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import ExceptionValue, TextValue

        v = ExceptionValue(
            nominal=NominalId(1),
            display_name="Abort",
            fields={"message": TextValue("fatal")},
        )
        out = render_value(v)
        assert out == 'Abort(message = "fatal")'
        assert "<dsl-value" not in out

    # ------------------------------------------------------------------
    # Nested text escaping including % → \%
    # ------------------------------------------------------------------

    def test_nested_text_escapes_percent(self) -> None:
        """Nested text containing ``%{`` escapes ``%`` as ``\\%``."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import ArrayValue, TextValue

        v = ArrayValue(elements=[TextValue("a%{b}")])
        out = render_value(v)
        assert out == r'["a\%{b}"]'

    def test_nested_text_escapes_quotes_and_newlines(self) -> None:
        """Nested text: quotes and newlines are escaped."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import ArrayValue, TextValue

        v = ArrayValue(elements=[TextValue('say "hi"\nbye')])
        out = render_value(v)
        assert out == r'["say \"hi\"\nbye"]'

    def test_quoted_top_level_text_escapes_percent(self) -> None:
        """render_value: top-level text with ``%`` escapes it when quoted."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import TextValue

        out = render_value(TextValue("a%{b}"), quote_strings=True)
        assert out == r'"a\%{b}"'

    # ------------------------------------------------------------------
    # pretty: single-line by default, indented when requested
    # ------------------------------------------------------------------

    def test_json_top_level_is_single_line_by_default(self) -> None:
        """Top-level json value renders as single-line JSON by default."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import JsonValue

        out = render_value(JsonValue({"k": 1}))
        assert out == '{"k": 1}'

    def test_json_top_level_is_pretty_when_requested(self) -> None:
        """Top-level json value renders as pretty-printed JSON when requested."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import JsonValue

        out = render_value(JsonValue({"k": 1}), pretty=True)
        assert out == '{\n  "k": 1\n}'

    def test_json_scalar_pretty_has_no_extra_indentation(self) -> None:
        """Pretty rendering of scalar json remains a single scalar."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import JsonValue

        assert render_value(JsonValue(1), pretty=True) == "1"

    def test_json_nested_is_compact(self) -> None:
        """json nested inside a record field renders compact by default."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import JsonValue, RecordValue

        v = RecordValue(
            nominal=NominalId(1),
            display_name="R",
            fields={"data": JsonValue({"a": 1, "b": 2})},
        )
        out = render_value(v)
        # The json field must be compact (no newlines) so the record stays single-line.
        assert "\n" not in out
        assert out == 'R(data = {"a": 1, "b": 2})'

    def test_array_pretty(self) -> None:
        """Pretty lists render over multiple indented lines."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import ArrayValue, IntValue

        v = ArrayValue(elements=[IntValue(1), IntValue(2)])
        assert render_value(v, pretty=True) == "[\n  1,\n  2\n]"

    def test_dict_pretty(self) -> None:
        """Pretty dicts render over multiple indented lines."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import DictValue, IntValue

        v = DictValue({"a": IntValue(1), "b": IntValue(2)})
        assert render_value(v, pretty=True) == '{\n  "a": 1,\n  "b": 2\n}'

    def test_record_pretty(self) -> None:
        """Pretty records render fields over multiple indented lines."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import IntValue, RecordValue, TextValue

        v = RecordValue(
            nominal=NominalId(1),
            display_name="Issue",
            fields={"title": TextValue("Missing tests"), "severity": IntValue(3)},
        )
        assert (
            render_value(v, pretty=True) == 'Issue(\n  title = "Missing tests",\n  severity = 3\n)'
        )

    def test_pretty_nested_indentation(self) -> None:
        """Pretty rendering indents nested structures recursively."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import ArrayValue, IntValue, RecordValue, TextValue

        v = RecordValue(
            nominal=NominalId(1),
            display_name="Issue",
            fields={
                "title": TextValue("Missing tests"),
                "scores": ArrayValue(elements=[IntValue(1), IntValue(2)]),
            },
        )
        assert (
            render_value(v, pretty=True)
            == 'Issue(\n  title = "Missing tests",\n  scores = [\n    1,\n    2\n  ]\n)'
        )

    # ------------------------------------------------------------------
    # display options: non-text values can use the same single renderer
    # ------------------------------------------------------------------

    def test_display_non_text_matches_default_for_scalar_values(self) -> None:
        """Display options do not alter scalar values."""
        from decimal import Decimal

        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import (
            ArrayValue,
            BoolValue,
            DecimalValue,
            IntValue,
            UnitValue,
        )

        for v in (
            IntValue(42),
            DecimalValue(Decimal("1.5")),
            BoolValue(True),
            UnitValue(),
            ArrayValue(elements=[IntValue(1)]),
        ):
            assert render_value(v, pretty=True, quote_strings=True) == render_value(v, pretty=True)

    def test_nested_text_in_array_is_quoted(self) -> None:
        """Text inside an array is quoted regardless of top-level quote mode."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import ArrayValue, TextValue

        out = render_value(
            ArrayValue(elements=[TextValue("v")]),
            quote_strings=False,
        )
        assert out == '["v"]'

    # ------------------------------------------------------------------
    # No <dsl-value> tags ever
    # ------------------------------------------------------------------

    def test_no_dsl_value_tags_in_interpolation(self) -> None:
        """Interpolation never wraps values in <dsl-value> tags."""
        from agm.agl.runtime.render import render_value
        from agm.agl.semantics.values import IntValue, TextValue

        assert "<dsl-value" not in render_value(TextValue("x"))
        assert "<dsl-value" not in render_value(IntValue(1))


# ---------------------------------------------------------------------------
# Coverage: serialize.py — value_to_json_obj and dumps_exact branches
# ---------------------------------------------------------------------------


class TestSerialize:
    """Coverage for serialize.py branches not exercised by higher-level tests."""

    def test_bool_value_serialized(self) -> None:
        from agm.agl.runtime.serialize import value_to_json_obj
        from agm.agl.semantics.values import BoolValue

        assert value_to_json_obj(BoolValue(True)) is True
        assert value_to_json_obj(BoolValue(False)) is False

    def test_dict_value_serialized(self) -> None:
        from agm.agl.runtime.serialize import value_to_json_obj
        from agm.agl.semantics.values import DictValue, IntValue

        result = value_to_json_obj(DictValue(entries={"a": IntValue(1)}))
        assert result == {"a": 1}

    def test_record_value_serialized(self) -> None:
        from agm.agl.runtime.serialize import value_to_json_obj
        from agm.agl.semantics.values import IntValue, RecordValue

        result = value_to_json_obj(
            RecordValue(nominal=NominalId(1), display_name="R", fields={"x": IntValue(5)})
        )
        assert result == {"x": 5}

    def test_enum_value_serialized(self) -> None:
        from agm.agl.runtime.serialize import value_to_json_obj
        from agm.agl.semantics.values import EnumValue, TextValue

        result = value_to_json_obj(
            EnumValue(
                nominal=NominalId(1),
                display_name="E",
                variant="A",
                fields={"msg": TextValue("hi")},
            )
        )
        assert result == {"$case": "A", "msg": "hi"}

    def test_pretty_array_serialized(self) -> None:
        from agm.agl.runtime.serialize import dumps_exact

        assert dumps_exact([1, 2], indent=2) == "[\n  1,\n  2\n]"

    def test_enum_nullary_value_serialized(self) -> None:
        from agm.agl.runtime.serialize import value_to_json_obj
        from agm.agl.semantics.values import EnumValue

        result = value_to_json_obj(
            EnumValue(nominal=NominalId(1), display_name="E", variant="Done", fields={})
        )
        assert result == {"$case": "Done"}

    def test_exception_value_serialized(self) -> None:
        from agm.agl.runtime.serialize import value_to_json_obj
        from agm.agl.semantics.values import ExceptionValue, TextValue

        result = value_to_json_obj(
            ExceptionValue(
                nominal=NominalId(1),
                display_name="Err",
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

    def test_dumps_exact_array_empty(self) -> None:
        from agm.agl.runtime.serialize import dumps_exact

        assert dumps_exact([]) == "[]"

    def test_dumps_exact_array_no_indent(self) -> None:
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


# ---------------------------------------------------------------------------
# Coverage: contract.py — ValueError when codec not found
# ---------------------------------------------------------------------------


class TestMaterializeContractMissingCodec:
    """Cover the ValueError branch when the codec is not in the registry."""

    def test_missing_codec_raises_value_error(self) -> None:
        from agm.agl.runtime.codec import TextCodec
        from agm.agl.runtime.contract import materialize_contract
        from agm.agl.semantics.types import TextType
        from agm.agl.typecheck.env import OutputContractSpec

        spec = OutputContractSpec(
            target_type=TextType(),
            codec_name="nonexistent_codec",
            strict_json=None,
        )
        with pytest.raises(ValueError, match="nonexistent_codec"):
            materialize_contract(spec, {"text": TextCodec()})


# ---------------------------------------------------------------------------
# params.py — host engine seeds and engine-key defaults
# ---------------------------------------------------------------------------


class TestEngineSettingDefaults:
    """The host side owns defaults only for the keys it can actually decode."""

    def test_unknown_engine_seed_is_rejected(self) -> None:
        from agm.agl.runtime.params import build_engine_config_seeds

        with pytest.raises(ValueError, match="unknown engine key"):
            build_engine_config_seeds({"unknown": True})

    def test_engine_default_settings_has_no_default_agent_floor(self) -> None:
        """``default-agent`` is declared by ``std/config``, not fabricated by the host."""
        from agm.agl.runtime.params import engine_default_settings

        assert "default-agent" not in engine_default_settings()


# ---------------------------------------------------------------------------
# Coverage: pipeline.py — generic exception handlers and error paths
# ---------------------------------------------------------------------------


class TestRuntimeErrorPaths:
    """Cover the generic exception handler branches in PipelineDriver.run."""

    def test_generic_parse_exception_covered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Generic (non-AglSyntaxError) exception in parse step → ok=False."""
        import agm.agl.modules.loader as parser_mod

        def bad_parse(*args: object, **kwargs: object) -> object:
            raise RuntimeError("unexpected parse error")

        monkeypatch.setattr(parser_mod, "parse_program_seeded", bad_parse)
        rt = PipelineDriver()
        result = run_inline_command(rt, "let x = 1")
        assert result.ok is False
        assert any("unexpected parse error" in d.message for d in result.diagnostics)

    def test_tab_warning_included_even_on_parse_failure(self) -> None:
        """Tab advisories come from the lexer's single scan, so they survive a
        parse failure: the scan completes (recording the TAB) before the grammar
        rejects the token stream."""
        rt = PipelineDriver()
        result = run_inline_command(rt, "\tprint")  # leading TAB, then an incomplete `print`
        assert result.ok is False
        assert result.diagnostics  # genuine parse error surfaced
        tab_warns = [w for w in result.warnings if w.severity == "warning"]
        assert len(tab_warns) == 1
        assert tab_warns[0].line == 1

    def test_generic_scope_exception_covered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Generic (non-AglScopeError) exception in scope step → ok=False."""
        import agm.agl.scope.program as scope_mod

        def bad_resolve(program: object) -> object:
            raise RuntimeError("unexpected scope error")

        monkeypatch.setattr(scope_mod, "resolve_program", bad_resolve)
        rt = PipelineDriver()
        result = run_inline_command(rt, "let x = 1")
        assert result.ok is False
        assert any("Scope error" in d.message for d in result.diagnostics)

    def test_generic_typecheck_exception_covered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Generic (non-AglTypeError) exception in typecheck step → ok=False."""
        import agm.agl.typecheck.program as tc_mod

        def bad_check(resolved: object, caps: object) -> object:
            raise RuntimeError("unexpected type error")

        monkeypatch.setattr(tc_mod, "check_program", bad_check)
        rt = PipelineDriver()
        result = run_inline_command(rt, "let x = 1")
        assert result.ok is False
        assert any("Type error" in d.message for d in result.diagnostics)

    def test_contract_error_returns_not_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Contract materialization error → ok=False with contract error diagnostic."""
        import agm.agl.pipeline as runtime_mod

        monkeypatch.setattr(
            runtime_mod,
            "_materialize_ir_contracts",
            lambda executable, codecs: (
                {},
                [Diagnostic(message="Contract error: bad contract", line=1)],
            ),
        )
        rt = PipelineDriver(agent_dispatcher=lambda req: "ok")
        result = run_inline_command(rt, 'ask "hi"')
        assert result.ok is False
        assert any("Contract error" in d.message for d in result.diagnostics)

    def test_uncaught_agl_raise_in_run(self) -> None:
        """AglRaise propagating from the interpreter → RunResult with error."""
        from agm.agl.semantics.exceptions import AglRaise
        from agm.agl.semantics.values import ExceptionValue, TextValue

        def bad_agent(req: object) -> str:
            raise AglRaise(
                ExceptionValue(
                    nominal=NominalId(1),
                    display_name="Abort",
                    fields={"message": TextValue("stopped")},
                )
            )

        rt = PipelineDriver(agent_dispatcher=bad_agent)
        result = run_inline_command(rt, 'ask "hi"')
        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "Abort"

    def test_text_param_not_str_raises(self) -> None:
        """convert_param_value: text type with non-str value → ValueError."""
        from agm.agl.runtime.params import convert_param_value
        from agm.agl.semantics.types import TextType

        with pytest.raises(ValueError, match="expected a text value"):
            convert_param_value("msg", 42, TextType(), type_table_for())

    def test_int_param_decimal_integral_widened(self) -> None:
        """convert_param_value: integral Decimal → IntValue for int type."""
        from decimal import Decimal

        from agm.agl.runtime.params import convert_param_value
        from agm.agl.semantics.types import IntType
        from agm.agl.semantics.values import IntValue

        result = convert_param_value("n", Decimal("3"), IntType(), type_table_for())
        assert result == IntValue(3)

    def test_int_param_non_integral_fails(self) -> None:
        """convert_param_value: non-integral value → ValueError for int type."""
        from agm.agl.runtime.params import convert_param_value
        from agm.agl.semantics.types import IntType

        with pytest.raises(ValueError, match="not of type 'integer'"):
            convert_param_value("n", "1.5", IntType(), type_table_for())

    def test_decimal_param_from_int(self) -> None:
        """convert_param_value: int value → DecimalValue for decimal type."""
        from decimal import Decimal

        from agm.agl.runtime.params import convert_param_value
        from agm.agl.semantics.types import DecimalType
        from agm.agl.semantics.values import DecimalValue

        result = convert_param_value("d", 3, DecimalType(), type_table_for())
        assert isinstance(result, DecimalValue)
        assert result.value == Decimal(3)

    def test_decimal_param_invalid_type_fails(self) -> None:
        """convert_param_value: bool value → ValueError for decimal type."""
        from agm.agl.runtime.params import convert_param_value
        from agm.agl.semantics.types import DecimalType

        with pytest.raises(ValueError, match="not of type 'number'"):
            convert_param_value("d", "true", DecimalType(), type_table_for())

    def test_bool_param_invalid_type_fails(self) -> None:
        """convert_param_value: non-bool value → ValueError for bool type."""
        from agm.agl.runtime.params import convert_param_value
        from agm.agl.semantics.types import BoolType

        with pytest.raises(ValueError, match="not of type 'boolean'"):
            convert_param_value("b", "1", BoolType(), type_table_for())

    def test_bool_param_true_succeeds(self) -> None:
        """convert_param_value: bool value → BoolValue for bool type."""
        from agm.agl.runtime.params import convert_param_value
        from agm.agl.semantics.types import BoolType
        from agm.agl.semantics.values import BoolValue

        result = convert_param_value("b", True, BoolType(), type_table_for())
        assert result == BoolValue(True)

    # --- assertions migrated from TestRuntimeExceptionHandlers (eval tests) ---

    def test_internal_interpreter_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """an unexpected (non-AglRaise) interpreter error must propagate.

        A Python-level bug must crash loudly rather than masquerade as a
        user-facing pre-execution diagnostic.
        """
        from agm.agl.eval.ir_interpreter import IrInterpreter

        def bad_execute(self: IrInterpreter, **_: object) -> dict[str, object]:
            raise RuntimeError("internal crash")

        monkeypatch.setattr(IrInterpreter, "run", bad_execute)

        rt = PipelineDriver()
        with pytest.raises(RuntimeError, match="internal crash"):
            run_inline_command(rt, "let x = 1\nx")

    def test_exception_value_to_run_error_maps_all_field_kinds(self) -> None:
        """exception_value_to_run_error converts every Value kind to JSON shape.

        This is the pure converter used to surface an uncaught AgL exception
        (e.g. AgentParseError) as a RunError.
        """
        import decimal

        from agm.agl.pipeline import RunError, exception_value_to_run_error
        from agm.agl.semantics.values import (
            ArrayValue,
            BoolValue,
            DecimalValue,
            DictValue,
            EnumValue,
            ExceptionValue,
            IntValue,
            JsonValue,
            RecordValue,
            TextValue,
        )

        exc_val = ExceptionValue(
            nominal=NominalId(1),
            display_name="AgentParseError",
            fields={
                "message": TextValue("failed"),
                "raw": TextValue("abc"),
                "agent": TextValue("ask"),
                "attempts": IntValue(1),
                "target_type": TextValue("text"),
                "decimal_val": DecimalValue(decimal.Decimal("1.5")),
                "bool_val": BoolValue(True),
                "json_val": JsonValue({"k": "v"}),
                "list_val": ArrayValue(elements=[IntValue(1)]),
                "dict_val": DictValue(entries={"x": IntValue(2)}),
                "rec_val": RecordValue(
                    nominal=NominalId(2),
                    display_name="R",
                    fields={"f": TextValue("v")},
                ),
                "enum_val": EnumValue(
                    nominal=NominalId(3), display_name="E", variant="V", fields={}
                ),
                "exc_val": ExceptionValue(nominal=NominalId(4), display_name="Inner", fields={}),
                "none_val": JsonValue(None),
            },
        )
        error = exception_value_to_run_error(exc_val)
        assert isinstance(error, RunError)
        assert error.type_name == "AgentParseError"
        assert error.fields["message"] == "failed"
        # Decimal is preserved exactly (not converted to float).
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
        from agm.agl.runtime.params import convert_param_value
        from agm.agl.semantics.types import JsonType
        from agm.agl.semantics.values import JsonValue

        result = convert_param_value("meta", [1, 2, 3], JsonType(), type_table_for())
        assert result == JsonValue([1, 2, 3])

    def test_convert_param_value_json_type_decodes_normalized_value(self) -> None:
        from agm.agl.runtime.params import convert_param_value
        from agm.agl.semantics.types import JsonType
        from agm.agl.semantics.values import JsonValue

        result = convert_param_value("meta", "1.0", JsonType(), type_table_for())

        assert result == JsonValue(1)
        assert isinstance(result.raw, int)

    def test_convert_param_value_array_type_parsed_via_json_codec(self) -> None:
        # array/dict/record/enum params are now accepted via the JsonCodec.
        from agm.agl.runtime.params import convert_param_value
        from agm.agl.semantics.types import ArrayType, TextType
        from agm.agl.semantics.values import ArrayValue, TextValue

        result = convert_param_value(
            "xs", '["a", "b"]', ArrayType(elem=TextType()), type_table_for()
        )
        assert isinstance(result, ArrayValue)
        assert result.elements == [TextValue("a"), TextValue("b")]

    def test_agl_raise_from_interpreter_becomes_run_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An AglRaise from the interpreter → RunResult.error (not None)."""
        from agm.agl.eval.ir_interpreter import IrInterpreter
        from agm.agl.semantics.exceptions import AglRaise
        from agm.agl.semantics.values import ExceptionValue, TextValue

        def bad_execute(self: IrInterpreter, **_: object) -> dict[str, object]:
            exc_val = ExceptionValue(
                nominal=NominalId(1),
                display_name="Abort",
                fields={"message": TextValue("fatal")},
            )
            raise AglRaise(exc_val)

        monkeypatch.setattr(IrInterpreter, "run", bad_execute)

        rt = PipelineDriver()
        result = run_inline_command(rt, "let x = 1\nx")
        assert result.ok is False
        assert result.error is not None
        assert result.error.type_name == "Abort"
        assert result.error.fields.get("message") == "fatal"

    # --- Structured Decimal params and non-JSON-shaped rejection ---

    def test_array_decimal_param_validates_exactly(self) -> None:
        """A array[decimal] param with native Decimal values must bind
        correctly without the old default=str corruption.

        Before the fix, Decimal("1.5") was serialized as the JSON string "1.5"
        (quoted), which failed schema validation.
        """
        import decimal as _decimal

        from agm.agl.runtime.params import convert_param_value
        from agm.agl.semantics.types import ArrayType, DecimalType
        from agm.agl.semantics.values import ArrayValue, DecimalValue

        result = convert_param_value(
            "xs",
            [_decimal.Decimal("1.5"), _decimal.Decimal("2.75")],
            ArrayType(elem=DecimalType()),
            type_table_for(),
        )
        assert isinstance(result, ArrayValue)
        assert result.elements == [
            DecimalValue(_decimal.Decimal("1.5")),
            DecimalValue(_decimal.Decimal("2.75")),
        ]

    def test_non_json_shaped_object_yields_clean_diagnostic(self) -> None:
        """A non-JSON-shaped object (e.g. a set) must yield a clean
        param-validation error naming the param, not a stringified value or
        traceback.
        """
        from agm.agl.runtime.params import convert_param_value
        from agm.agl.semantics.types import ArrayType, TextType

        with pytest.raises(ValueError, match="xs") as exc_info:
            convert_param_value("xs", {1, 2, 3}, ArrayType(elem=TextType()), type_table_for())
        # The error message must name the param and mention the type, not
        # contain a raw repr of the set or a json.dumps traceback.
        msg = str(exc_info.value)
        assert "set" in msg  # type name named

    def test_decimal_native_in_array_end_to_end(self, capsys: pytest.CaptureFixture[str]) -> None:
        """E2e: param xs: array[decimal] with Decimal values binds and prints."""
        import decimal as _decimal

        result = run_inline_command(
            PipelineDriver(),
            "param xs: array[decimal]\nprint xs\n",
            param_values={"xs": [_decimal.Decimal("1.5"), _decimal.Decimal("2.25")]},
        )
        assert result.ok is True
        out = capsys.readouterr().out
        assert "1.5" in out
        assert "2.25" in out

    def test_float_native_in_decimal_array_end_to_end(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Native JSON-shaped floats are canonicalized before typed decoding."""
        result = run_inline_command(
            PipelineDriver(),
            "param xs: array[decimal]\nprint xs\n",
            param_values={"xs": [1.5, 2.25]},
        )

        assert result.ok is True
        assert result.diagnostics == []
        assert capsys.readouterr().out == "[1.5, 2.25]\n"

    def test_render_builtin_returns_single_line_unquoted_text(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = run_inline_command(
            PipelineDriver(),
            'print(render("hello", quote_strings = false))\n'
            "print(render([1, 2], pretty = false))\n"
            'print(render({"a": 1} as json, pretty = false))\n',
        )

        assert result.ok is True
        assert result.diagnostics == []
        assert capsys.readouterr().out == 'hello\n[1, 2]\n{"a": 1}\n'

    def test_render_builtin_accepts_render_options(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = run_inline_command(
            PipelineDriver(),
            'print(render("hello"))\nprint(render([1, 2]))\nprint(render({"a": 1} as json))\n',
        )

        assert result.ok is True
        assert result.diagnostics == []
        assert capsys.readouterr().out == '"hello"\n[\n  1,\n  2\n]\n{\n  "a": 1\n}\n'

    def test_is_json_shaped_dict_with_non_str_key_is_false(self) -> None:
        """_is_json_shaped: a dict with non-str keys is not JSON-shaped (covers
        the dict branch of _is_json_shaped).
        """
        from agm.agl.runtime.params import _is_json_shaped

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

        rt = PipelineDriver(agent_dispatcher=agent)
        result = run_inline_command(
            rt,
            'param x\nask("see: %{x}")',
            param_values={"x": "hello"},
        )
        assert result.ok is True
        assert received, "agent should have been called"
        prompt = received[0].prompt
        assert "hello" in prompt
        assert "<dsl-value" not in prompt

    def test_array_interpolation_in_prompt_is_pretty_json(self) -> None:
        received: list[AgentRequest] = []

        def agent(req: AgentRequest) -> str:
            received.append(req)
            return "ok"

        rt = PipelineDriver(agent_dispatcher=agent)
        result = run_inline_command(
            rt,
            'let items: array[text] = ["a", "b"]\nask("items: %{items}")',
        )
        assert result.ok is True
        prompt = received[0].prompt
        assert "a" in prompt
        assert "b" in prompt
        assert "<dsl-value" not in prompt


class TestMaxIterationsExceededSchema:
    """``MaxIterationsExceeded`` carries the full .

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

    def test_fields_surface_through_real_source(self, capsys: pytest.CaptureFixture[str]) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, self._PROGRAM)
        # The exception is caught, so the run completes successfully.
        assert result.ok is True
        assert result.error is None
        lines = capsys.readouterr().out.splitlines()
        # limit, condition (exact until-expression source), last_condition_value.
        assert lines == ["2", "n > 10", "false"]

    def test_metadata_field_is_accessible(self, capsys: pytest.CaptureFixture[str]) -> None:
        # ``metadata`` is a json placeholder, but
        # it is part of the schema and must be readable as a field.
        rt = PipelineDriver()
        program = (
            "try\n"
            "  do[1] ()\n"
            "  until false\n"
            "catch MaxIterationsExceeded as e =>\n"
            "  print e.metadata\n"
        )
        result = run_inline_command(rt, program)
        assert result.ok is True
        assert capsys.readouterr().out.strip() == "null"

    def test_condition_reflects_each_distinct_until_expression(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A different until-expression must yield a different ``condition`` slice,
        # proving the source text is recovered per-node rather than hard-coded.
        rt = PipelineDriver()
        program = (
            "let finished = false\n"
            "try\n"
            "  do[1] ()\n"
            "  until finished\n"
            "catch MaxIterationsExceeded as e =>\n"
            "  print e.condition\n"
            "  print e.last_condition_value\n"
        )
        result = run_inline_command(rt, program)
        assert result.ok is True
        lines = capsys.readouterr().out.splitlines()
        assert lines == ["finished", "false"]


class TestExhaustivenessErrorSurfaces:
    """A non-exhaustive enum ``case`` is rejected before execution."""

    def test_error_surfaces_and_run_does_not_execute(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rt = PipelineDriver()
        program = (
            'enum R\n  | Pass\n  | Fail\nlet r: R = Pass()\ncase r of\n  | Pass() => print "ok"\n'
        )
        result = run_inline_command(rt, program)
        assert result.ok is False
        assert result.error is None
        assert len(result.diagnostics) == 1
        assert result.diagnostics[0].severity == "error"
        assert "Fail" in result.diagnostics[0].message
        assert result.warnings == []
        assert capsys.readouterr().out == ""


# Permission-based tests: chmod 0o444 has no effect for root, who can write
# regardless.  Skip there rather than assert a false negative.
_skip_if_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="permission tests are meaningless as root (root bypasses file modes)",
)


class TestTraceWriteFailureIsBestEffort:
    """a mid-run trace write failure must not corrupt program semantics."""

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
        rt = PipelineDriver()

        # Pre-create the trace file and make it read-only so the first record
        # write (run_start) fails — the run must still complete normally.
        log_file.write_text("")
        log_file.chmod(0o444)

        program = 'print "a"\nprint "b"\nprint "c"\n'
        result = run_inline_command(rt, program, log_file=log_file)

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
        rt = PipelineDriver()
        result = run_inline_command(rt, "let x = 1")
        tab_warns = [w for w in result.warnings if "TAB" in w.message]
        assert tab_warns == []

    def test_tab_in_valid_source_yields_warning(self) -> None:
        # TAB used as whitespace inside a valid statement on the second line.
        source = "let x = 1\nlet\ty = 2"
        rt = PipelineDriver()
        result = run_inline_command(rt, source)
        tab_warns = [w for w in result.warnings if "TAB" in w.message]
        assert len(tab_warns) == 1
        assert tab_warns[0].line == 2

    def test_tab_warning_does_not_affect_ok(self) -> None:
        # A tab warning must not cause ok to become False.
        source = "let\tx = 1\nx"
        rt = PipelineDriver()
        result = run_inline_command(rt, source)
        assert result.ok is True
        assert result.error is None

    def test_multiple_tabs_multiple_warnings(self) -> None:
        source = "let\tx = 1\nlet\ty = 2"
        rt = PipelineDriver()
        result = run_inline_command(rt, source)
        tab_warns = [w for w in result.warnings if "TAB" in w.message]
        assert len(tab_warns) == 2


# ---------------------------------------------------------------------------
# declared_agents() API
# ---------------------------------------------------------------------------


class TestDeclaredAgentsApi:
    """PipelineDriver.declared_agents(): parse + scope only, non-raising."""


# ---------------------------------------------------------------------------
# Source↔host reconciliation in run()
# ---------------------------------------------------------------------------


class TestLegacyAgentRegistry:
    """Legacy registrations are optional bridges for command-valued agents."""

    def test_command_value_dispatches_through_a_matching_registration(self) -> None:
        calls: list[str] = []

        def agent(req: AgentRequest) -> str:
            calls.append(req.prompt)
            return "output"

        rt = PipelineDriver(agent_dispatcher=agent)
        result = run_inline_command(
            rt, 'let impl = AgentCommand("impl")\nask("do it", agent = impl)'
        )
        assert result.ok
        assert calls == ["do it"]

    def test_command_value_uses_the_default_dispatcher(self) -> None:
        rt = PipelineDriver(agent_dispatcher=lambda req: "ok")
        result = run_inline_command(
            rt, 'let any_name = AgentCommand("any-name")\nask("hi", agent = any_name)'
        )
        assert result.ok


# ---------------------------------------------------------------------------
# Coverage: schema.py — derive_schema branches not exercised higher up
# ---------------------------------------------------------------------------


class TestDeriveSchema:
    """Unit tests for derive_schema covering all type branches."""

    def test_bool_type(self) -> None:
        from agm.agl.semantics.types import BoolType
        from agm.agl.type_schema import derive_schema

        assert derive_schema(BoolType(), type_table_for()) == {"type": "boolean"}

    def test_json_type(self) -> None:
        from agm.agl.semantics.types import JsonType
        from agm.agl.type_schema import derive_schema

        assert derive_schema(JsonType(), type_table_for()) == {}

    def test_dict_type(self) -> None:
        from agm.agl.semantics.types import DictType, IntType
        from agm.agl.type_schema import derive_schema

        result = derive_schema(DictType(value=IntType()), type_table_for())
        assert result == {"type": "object", "additionalProperties": {"type": "integer"}}

    def test_record_type(self) -> None:
        from agm.agl.semantics.types import TextType
        from agm.agl.type_schema import derive_schema
        from tests._agl_helpers import record_type

        typ, typedef = record_type("Point", {"x": TextType()})
        result = derive_schema(typ, type_table_for(typedef))
        assert result["type"] == "object"
        assert result["required"] == ["x"]
        assert result["additionalProperties"] is False

    def test_enum_type_with_payload(self) -> None:
        from agm.agl.semantics.types import TextType
        from agm.agl.type_schema import derive_schema
        from tests._agl_helpers import enum_type

        typ, typedef = enum_type("Status", {"Pass": {}, "Fail": {"reason": TextType()}})
        result = derive_schema(typ, type_table_for(typedef))
        assert "oneOf" in result
        assert len(result["oneOf"]) == 2

    def test_exception_type_raises(self) -> None:
        from agm.agl.semantics.types import ExceptionType
        from agm.agl.type_schema import derive_schema

        with pytest.raises(TypeError, match="ExceptionType"):
            derive_schema(ExceptionType(name="MyErr"), type_table_for())

    def test_unit_type_raises(self) -> None:
        from agm.agl.semantics.types import UnitType
        from agm.agl.type_schema import derive_schema

        with pytest.raises(TypeError, match="UnitType"):
            derive_schema(UnitType(), type_table_for())

    def test_function_type_raises(self) -> None:
        from agm.agl.semantics.types import FunctionType, TextType
        from agm.agl.type_schema import derive_schema

        with pytest.raises(TypeError, match="FunctionType"):
            derive_schema(FunctionType(params=(TextType(),), result=TextType()), type_table_for())

    def test_bottom_type_raises(self) -> None:
        from agm.agl.semantics.types import BottomType
        from agm.agl.type_schema import derive_schema

        with pytest.raises(TypeError, match="BottomType"):
            derive_schema(BottomType(), type_table_for())


# ---------------------------------------------------------------------------
# Coverage: type_schema.py — build_param_decoder and build_format_instructions
# ---------------------------------------------------------------------------


class TestBuildParamDecoder:
    """Direct tests for build_param_decoder (previously zero direct references)."""

    def test_text_type_is_verbatim(self) -> None:
        """TextType params are taken verbatim — text_verbatim is True."""
        from agm.agl.semantics.types import TextType
        from agm.agl.type_schema import build_param_decoder

        decoder = build_param_decoder(TextType(), type_table_for())
        assert decoder.text_verbatim is True

    def test_text_type_target_label(self) -> None:
        """target_type_label is repr(TextType())."""
        from agm.agl.semantics.types import TextType
        from agm.agl.type_schema import build_param_decoder

        decoder = build_param_decoder(TextType(), type_table_for())
        assert decoder.target_type_label == repr(TextType())

    def test_int_type_not_verbatim(self) -> None:
        """Non-text types are NOT verbatim."""
        from agm.agl.semantics.types import IntType
        from agm.agl.type_schema import build_param_decoder

        decoder = build_param_decoder(IntType(), type_table_for())
        assert decoder.text_verbatim is False

    def test_int_type_json_schema_matches_derive_schema(self) -> None:
        """json_schema is json.dumps(derive_schema(typ), sort_keys=True)."""
        import json

        from agm.agl.semantics.types import IntType
        from agm.agl.type_schema import build_param_decoder, derive_schema

        typ = IntType()
        table = type_table_for()
        decoder = build_param_decoder(typ, table)
        expected_schema = json.dumps(derive_schema(typ, table), sort_keys=True)
        assert decoder.json_schema == expected_schema
        # Parses as valid JSON
        parsed = json.loads(decoder.json_schema)
        assert parsed == {"type": "integer"}

    def test_record_type_json_schema_matches_derive_schema(self) -> None:
        """A record type's json_schema embeds the full record schema."""
        import json

        from agm.agl.semantics.types import TextType
        from agm.agl.type_schema import build_param_decoder, derive_schema
        from tests._agl_helpers import record_type

        typ, typedef = record_type("Point", {"x": TextType()})
        table = type_table_for(typedef)
        decoder = build_param_decoder(typ, table)
        expected_schema = json.dumps(derive_schema(typ, table), sort_keys=True)
        assert decoder.json_schema == expected_schema
        # Sanity: the schema is valid JSON with expected structure
        parsed = json.loads(decoder.json_schema)
        assert parsed["type"] == "object"
        assert "x" in parsed["required"]

    def test_int_type_target_label_is_repr(self) -> None:
        """target_type_label is repr(typ) for int."""
        from agm.agl.semantics.types import IntType
        from agm.agl.type_schema import build_param_decoder

        typ = IntType()
        decoder = build_param_decoder(typ, type_table_for())
        assert decoder.target_type_label == repr(typ)

    def test_undecodable_type_raises_type_error(self) -> None:
        """Unit/agent/exception types raise TypeError (via derive_schema)."""
        from agm.agl.semantics.types import UnitType
        from agm.agl.type_schema import build_param_decoder

        with pytest.raises(TypeError):
            build_param_decoder(UnitType(), type_table_for())

    def test_exception_type_raises_type_error(self) -> None:
        from agm.agl.semantics.types import ExceptionType
        from agm.agl.type_schema import build_param_decoder

        with pytest.raises(TypeError):
            build_param_decoder(ExceptionType(name="MyErr"), type_table_for())


class TestBuildFormatInstructions:
    """Direct tests for build_format_instructions (previously zero direct references)."""

    def test_empty_schema_returns_no_schema_message(self) -> None:
        """An empty schema dict produces the 'Return exactly one JSON value.' message."""
        from agm.agl.type_schema import build_format_instructions

        result = build_format_instructions({})
        assert "Return exactly one JSON value." in result

    def test_empty_schema_no_code_fence(self) -> None:
        """Empty schema message must NOT contain a JSON code fence."""
        from agm.agl.type_schema import build_format_instructions

        result = build_format_instructions({})
        assert "```json" not in result

    def test_non_empty_schema_embeds_json_fence(self) -> None:
        """A non-empty schema dict produces output with a ```json code fence."""
        from agm.agl.type_schema import build_format_instructions

        schema: dict[str, object] = {"type": "object", "required": ["x"]}
        result = build_format_instructions(schema)
        assert "```json" in result

    def test_non_empty_schema_embeds_schema_content(self) -> None:
        """The schema JSON is embedded verbatim in the output."""
        import json

        from agm.agl.type_schema import build_format_instructions

        schema: dict[str, object] = {"type": "integer"}
        result = build_format_instructions(schema)
        schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
        assert schema_text in result

    def test_non_empty_schema_contains_json_value_instruction(self) -> None:
        """Non-empty output also says 'Return exactly one JSON value conforming…'."""
        from agm.agl.type_schema import build_format_instructions

        result = build_format_instructions({"type": "string"})
        assert "Return exactly one JSON value" in result


# ---------------------------------------------------------------------------
# Coverage: serialize.py — AglNonDataValue branches
# ---------------------------------------------------------------------------


class TestSerializeV2OpaqueValues:
    """Unit, agent, constructor, function, and iterator values have no JSON
    representation — each raises :class:`AglNonDataValue` with the matching
    user-facing ``kind``."""

    def test_unit_value_raises(self) -> None:
        from agm.agl.runtime.serialize import AglNonDataValue, value_to_json_obj
        from agm.agl.semantics.values import UnitValue

        with pytest.raises(AglNonDataValue) as exc_info:
            value_to_json_obj(UnitValue())
        assert exc_info.value.kind == "unit"

    def test_constructor_value_raises(self) -> None:
        from agm.agl.runtime.serialize import AglNonDataValue, value_to_json_obj
        from agm.agl.semantics.values import ConstructorValue

        ctor = ConstructorValue(nominal=NominalId(1), display_name="Box", variant=None)
        with pytest.raises(AglNonDataValue) as exc_info:
            value_to_json_obj(ctor)
        assert exc_info.value.kind == "constructor"

    def test_ir_closure_value_raises(self) -> None:
        """IrClosureValue has no JSON representation — AglNonDataValue is raised."""
        from agm.agl.ir.ids import FunctionId
        from agm.agl.runtime.serialize import AglNonDataValue, value_to_json_obj
        from agm.agl.semantics.values import IrClosureValue

        ir_closure = IrClosureValue(function_id=FunctionId(0), captures=())
        with pytest.raises(AglNonDataValue) as exc_info:
            value_to_json_obj(ir_closure)
        assert exc_info.value.kind == "function"

    def test_iterator_value_raises(self) -> None:
        from agm.agl.runtime.serialize import AglNonDataValue, value_to_json_obj
        from agm.agl.semantics.values import IteratorValue

        with pytest.raises(AglNonDataValue) as exc_info:
            value_to_json_obj(IteratorValue(elements=[]))
        assert exc_info.value.kind == "iterator"

    def test_marker_text_matches_kind(self) -> None:
        """``degraded_marker`` produces the same text every degrade site relies on."""
        from agm.agl.runtime.serialize import AglNonDataValue, degraded_marker

        exc = AglNonDataValue("unit")
        assert degraded_marker(exc) == "<unit has no JSON representation>"


# ---------------------------------------------------------------------------
# Coverage: pipeline.py — uncovered branches and current properties
# ---------------------------------------------------------------------------


class TestIrHostMetadataCoverage:
    def test_invalid_external_param_shapes_are_diagnostics(self) -> None:
        text_result = run_inline_command(
            PipelineDriver(), "param value: text\nprint value", param_values={"value": 3}
        )
        assert not text_result.ok
        assert "text value" in text_result.diagnostics[0].message

        json_result = run_inline_command(
            PipelineDriver(), "param value: int\nprint value", param_values={"value": {1, 2}}
        )
        assert not json_result.ok
        assert "JSON-compatible" in json_result.diagnostics[0].message

    def test_missing_ir_codec_materialization_is_diagnostic(self) -> None:
        from agm.agl.ir.contracts import ContractRequest
        from agm.agl.ir.ids import ContractId
        from agm.agl.ir.program import ExecutableProgram
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.runtime.params import _materialize_ir_contracts

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

    def test_integral_decimal_decodes_to_int(self) -> None:
        from decimal import Decimal

        from agm.agl.eval.conversions import decode_value
        from agm.agl.ir.contracts import ScalarDecode, ScalarKind
        from agm.agl.semantics.values import IntValue

        assert decode_value(ScalarDecode(ScalarKind.INT), Decimal("2.0")) == IntValue(2)


class TestRunErrorToMessage:
    def test_to_message_includes_message_and_location(self) -> None:
        from agm.agl.pipeline import RunError

        err = RunError(
            type_name="AgentParseError",
            fields={"message": "bad output"},
            line=5,
            col=3,
        )
        msg = err.to_message()
        assert "bad output" in msg
        assert "at line 5, col 3" in msg

    def test_to_message_includes_line_without_column(self) -> None:
        from agm.agl.pipeline import RunError

        err = RunError(
            type_name="SomeError",
            fields={"message": "oops"},
            line=2,
        )
        msg = err.to_message()
        assert "at line 2" in msg


class TestHostEnvironmentCache:
    """PipelineDriver.host_environment() caches and is invalidated on registration."""

    def test_host_environment_returns_same_object_on_second_call(self) -> None:
        rt = PipelineDriver()
        env1 = rt.host_environment()
        env2 = rt.host_environment()
        assert env1 is env2


class TestRegisterCodecErrors:
    """register_codec raises for reserved names and duplicates."""

    def _make_codec(self, name: str) -> OutputCodec:
        from agm.agl.runtime.codec import OutputCodec, ParseResult, TextCodec
        from agm.agl.runtime.contract import OutputContract
        from agm.agl.semantics.type_table import TypeTable
        from agm.agl.semantics.types import TextType
        from agm.agl.semantics.values import TextValue

        class _Codec(OutputCodec):
            @property
            def name(self) -> str:
                return _name

            @property
            def supported_kinds(self) -> frozenset[str]:
                return frozenset({"text"})

            def supports_type(self, t: Type) -> bool:
                return isinstance(t, TextType)

            def make_contract(
                self, type_ref: Type, type_table: TypeTable | None = None
            ) -> OutputContract:
                return OutputContract(
                    target_type_label=repr(type_ref),
                    codec=TextCodec(),
                    strict_json=None,
                    format_instructions="",
                    json_schema=None,
                )

            def parse(
                self,
                raw: str,
                *,
                strict_json: bool = False,
                schema: dict[str, object] | None = None,
                decode: object | None = None,
            ) -> ParseResult:
                return ParseResult.success(TextValue(""))

        _name = name
        return _Codec()

    def test_reserved_name_raises(self) -> None:
        rt = PipelineDriver()
        codec = self._make_codec("text")  # "text" is a builtin codec name
        with pytest.raises(ValueError, match="reserved"):
            rt.register_codec(codec)

    def test_duplicate_name_raises(self) -> None:
        rt = PipelineDriver()
        codec1 = self._make_codec("mycodec")
        codec2 = self._make_codec("mycodec")
        rt.register_codec(codec1)
        with pytest.raises(ValueError, match="already registered"):
            rt.register_codec(codec2)


class TestDefaultCallDepthLimit:
    """default_call_depth_limit constructor parameter and property."""

    def test_default_is_256(self) -> None:
        rt = PipelineDriver()
        assert rt.default_call_depth_limit == 256

    def test_custom_value_is_observable(self) -> None:
        rt = PipelineDriver(default_call_depth_limit=128)
        assert rt.default_call_depth_limit == 128


class TestConvertInputUnsupportedType:
    """convert_param_value raises ValueError for unsupported types (e.g. ArrayType of records)."""


# ---------------------------------------------------------------------------
# New Feature tests: user-defined functions, ExecResult, ask with AgentValue
# ---------------------------------------------------------------------------


class TestV2UserDefinedFunctions:
    """Def expressions: first-class functions, recursion, call depth limit."""

    def test_def_call_basic(self) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, "def add(a: int, b: int) -> int = a + b\nadd(1, 2)\n")
        assert result.ok is True

    def test_def_recursive_call(self) -> None:
        rt = PipelineDriver()
        result = run_inline_command(
            rt,
            "def fact(n: int) -> int =\n"
            "  if n <= 1 =>\n"
            "    1\n"
            "  | else =>\n"
            "    n * fact(n - 1)\n"
            "fact(5)\n",
        )
        assert result.ok is True

    def test_def_call_depth_limit_enforced(self) -> None:
        """Exceeding max_call_depth raises a RecursionError."""
        rt = PipelineDriver(default_call_depth_limit=10)
        result = run_inline_command(rt, "def inf(n: int) -> int =\n  inf(n + 1)\ninf(0)\n")
        assert result.ok is False


class TestV2ExecStructuredForm:
    """Structured exec form: let x: T = exec ... raises on nonzero."""

    def test_exec_text_form_captures_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, 'let out: text = exec "echo hello"\nprint out\n')
        assert result.ok is True
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_exec_nonzero_raises_when_typed(self) -> None:
        rt = PipelineDriver()
        result = run_inline_command(rt, 'let out: text = exec "false"\nprint out\n')
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

        rt = PipelineDriver(agent_dispatcher=agent)
        result = run_inline_command(
            rt, 'let helper = AgentCommand("helper")\nask("question", agent = helper)\n'
        )
        assert result.ok is True
        assert received == ["question"]


# ---------------------------------------------------------------------------
# PreparedProgram / prepare_program / run_prepared / discover_params
# ---------------------------------------------------------------------------


class TestPrepareProgram:
    """PipelineDriver.prepare_program: program front-end."""

    def test_prepare_program_no_imports_returns_prepared_program(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A module program with no imports produces a valid PreparedProgram."""
        from agm.agl.modules.roots import RootSet
        from agm.agl.pipeline import PreparedProgram

        roots = RootSet(roots=frozenset({_STDLIB_ROOT}))
        prepared = prepare_inline_command("let x = 1\nx", entry_path=None, roots=roots)
        assert isinstance(prepared, PreparedProgram)
        assert prepared.resolved is not None
        assert prepared.diagnostics == ()

    def test_prepare_program_syntax_error_captured(self, tmp_path: pathlib.Path) -> None:
        """A syntax error is captured as a diagnostic, not raised."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset({_STDLIB_ROOT}))
        prepared = prepare_inline_command("let x = !!!", entry_path=None, roots=roots)
        assert prepared.resolved is None
        assert len(prepared.diagnostics) >= 1

    def test_prepare_program_missing_import_captured(self, tmp_path: pathlib.Path) -> None:
        """A missing import is captured as a diagnostic, not raised."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset({_STDLIB_ROOT}))
        prepared = prepare_inline_command(
            "open import nonexistent/module\nlet x = 1\nx",
            entry_path=None,
            roots=roots,
        )
        assert prepared.resolved is None
        assert len(prepared.diagnostics) >= 1
        assert "nonexistent/module" in prepared.diagnostics[0].message

    def test_prepare_program_with_valid_import(self, tmp_path: pathlib.Path) -> None:
        """A valid import resolves when the module file exists on disk."""
        from agm.agl.modules.roots import RootSet

        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "mymod.agl").write_text("def add(a: int, b: int) -> int = a + b\n")

        roots = RootSet(roots=frozenset({lib_dir, _STDLIB_ROOT}))
        entry = "open import mymod\nlet r = add(2, 3)\nr"
        prepared = prepare_inline_command(entry, entry_path=None, roots=roots)
        assert prepared.resolved is not None
        assert prepared.diagnostics == ()

    def test_prepare_program_captures_syntax_error(self) -> None:
        """A parse failure is captured as a diagnostic, not raised."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset({_STDLIB_ROOT}))
        prepared = prepare_inline_command("let x = = =", entry_path=None, roots=roots)
        assert prepared.resolved is None
        assert prepared.diagnostics


class TestRunPreparedProgram:
    """PipelineDriver.run_prepared: graph execution."""

    def test_single_entry_graph_behaves_like_run(self, tmp_path: pathlib.Path) -> None:
        """A module program via run_prepared returns same result as run()."""
        from agm.agl.modules.roots import RootSet

        rt = PipelineDriver()
        roots = RootSet(roots=frozenset({_STDLIB_ROOT}))
        prepared = prepare_inline_command("let x = 1\nx", entry_path=None, roots=roots)
        result = rt.run_prepared(prepared)
        assert result.ok is True
        assert result.error is None

    def test_graph_with_library_module_executes(self, tmp_path: pathlib.Path) -> None:
        """A two-module graph (entry + library) runs to completion."""
        from agm.agl.modules.roots import RootSet

        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "mymod.agl").write_text("def add(a: int, b: int) -> int = a + b\n")

        roots = RootSet(roots=frozenset({lib_dir.resolve(), _STDLIB_ROOT}))
        entry = "open import mymod\nlet r = add(2, 3)\nr"
        prepared = prepare_inline_command(entry, entry_path=None, roots=roots)
        rt = PipelineDriver()
        result = rt.run_prepared(prepared)
        assert result.ok is True
        assert result.error is None

    def test_graph_failure_propagated(self, tmp_path: pathlib.Path) -> None:
        """When the load phase captured a scope error, run_prepared reports it."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset({_STDLIB_ROOT}))
        prepared = prepare_inline_command("let x = undefined_name", entry_path=None, roots=roots)
        rt = PipelineDriver()
        result = rt.run_prepared(prepared)
        assert result.ok is False
        assert result.error is None
        assert len(result.diagnostics) >= 1

    def test_graph_missing_import_propagated(self, tmp_path: pathlib.Path) -> None:
        """A missing import error from prepare_program flows through run_prepared."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset({_STDLIB_ROOT}))
        prepared = prepare_inline_command(
            "open import missing/module\nlet x = 1\nx", entry_path=None, roots=roots
        )
        rt = PipelineDriver()
        result = rt.run_prepared(prepared)
        assert result.ok is False
        assert "missing/module" in result.diagnostics[0].message

    def test_graph_check_only_returns_call_inventory(self, tmp_path: pathlib.Path) -> None:
        """check_only=True produces call_sites from the entry module."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset({_STDLIB_ROOT}))
        prepared = prepare_inline_command(
            'let r = ask("hello")\nprint r', entry_path=None, roots=roots
        )
        rt = PipelineDriver(agent_dispatcher=lambda req: "x")  # type: ignore[arg-type]
        result = rt.run_prepared(prepared, check_only=True)
        assert result.ok is True
        assert len(result.call_sites) >= 1

    def test_multimodule_wildcard_import(self, tmp_path: pathlib.Path) -> None:
        """Wildcard import brings multiple modules into scope."""
        from agm.agl.modules.roots import RootSet

        lib_dir = tmp_path / "lib"
        utils_dir = lib_dir / "utils"
        utils_dir.mkdir(parents=True)
        (utils_dir / "math.agl").write_text("def add(a: int, b: int) -> int = a + b\n")
        (utils_dir / "strings.agl").write_text(
            'def greet(name: text) -> text = "Hello, " + name + "!"\n'
        )

        roots = RootSet(roots=frozenset({lib_dir.resolve(), _STDLIB_ROOT}))
        entry = 'open import utils/*\nlet n = add(2, 3)\nlet g = greet("World")\nprint n\nprint g\n'
        prepared = prepare_inline_command(entry, entry_path=None, roots=roots)
        rt = PipelineDriver()
        result = rt.run_prepared(prepared, check_only=True)
        assert result.ok is True

    def test_multimodule_qualified_import(self, tmp_path: pathlib.Path) -> None:
        """Qualified import requires :: qualifier to access names."""
        from agm.agl.modules.roots import RootSet

        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        (lib_dir / "calc.agl").write_text("def square(n: int) -> int = n * n\n")

        roots = RootSet(roots=frozenset({lib_dir.resolve(), _STDLIB_ROOT}))
        entry = "import calc\nlet r = calc::square(5)\nr"
        prepared = prepare_inline_command(entry, entry_path=None, roots=roots)
        rt = PipelineDriver()
        result = rt.run_prepared(prepared)
        assert result.ok is True


class TestDiscoverParamsGraph:
    """PipelineDriver.discover_params: typed param discovery."""

    def test_discover_params_no_params(self, tmp_path: pathlib.Path) -> None:
        """discover_params returns empty params for a program with no params."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset({_STDLIB_ROOT}))
        prepared = prepare_inline_command("let x = 1\nx", entry_path=None, roots=roots)
        rt = PipelineDriver()
        discovery = rt.discover_params(prepared)
        assert discovery.diagnostics == ()
        assert discovery.params == ()

    def test_discover_params_with_declared_param(self, tmp_path: pathlib.Path) -> None:
        """discover_params discovers typed param declarations."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset({_STDLIB_ROOT}))
        prepared = prepare_inline_command(
            "param name: text\nprint name",
            entry_path=None,
            roots=roots,
        )
        rt = PipelineDriver()
        discovery = rt.discover_params(prepared)
        assert discovery.diagnostics == ()
        assert len(discovery.params) == 1
        assert discovery.params[0].name == "name"
        assert discovery.params[0].is_entry

    def test_discover_programs_records_module_and_paths(self, tmp_path: pathlib.Path) -> None:
        from agm.agl.modules.roots import RootSet

        (tmp_path / "helper.agl").write_text("program def helper() -> unit = ()\n")
        source = (
            "import helper\n"
            "scope review\n"
            "program def main() -> unit = ()\n"
            "end review\n"
            "program def top() -> unit = ()\n"
        )
        prepared = PipelineDriver.prepare_program(
            source,
            entry_path=tmp_path / "entry.agl",
            roots=RootSet(roots=frozenset({tmp_path})),
            default_stdlib=False,
        )

        discovery = PipelineDriver().discover_params(prepared)

        assert discovery.diagnostics == ()
        assert [
            (program.module.display(), program.scope_path, program.name, program.declaration_path)
            for program in discovery.programs
        ] == [
            ("<entry>", ("review",), "main", "review::main"),
            ("<entry>", (), "top", "top"),
            ("helper", (), "helper", "helper"),
        ]
        assert discovery.programs[0].qualified_path == "review::main"
        assert discovery.programs[-1].qualified_path == "helper::helper"

    def test_program_inventory_follows_loader_export_edges_only(
        self, tmp_path: pathlib.Path
    ) -> None:
        from agm.agl.modules.roots import RootSet

        (tmp_path / "runner.agl").write_text("export settings\nprogram def main() -> unit = ()\n")
        (tmp_path / "settings.agl").write_text('param token: text = "ok"\n')
        (tmp_path / "unrelated.agl").write_text('param ignored: text = "no"\n')
        prepared = PipelineDriver.prepare_program(
            "import runner\nimport unrelated\nprogram def entry() -> unit = ()\n",
            entry_path=tmp_path / "entry.agl",
            roots=RootSet(roots=frozenset({tmp_path})),
            default_stdlib=False,
        )

        discovery = PipelineDriver().discover_params(prepared)

        runner = next(
            program for program in discovery.programs if program.module.path_str() == "runner"
        )
        assert [param.name for param in discovery.params_for(runner)] == ["token"]

    def test_discover_params_failure_returns_diagnostics(self, tmp_path: pathlib.Path) -> None:
        """discover_params returns diagnostics when the prepare phase failed."""
        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset({_STDLIB_ROOT}))
        prepared = prepare_inline_command(
            "import no_such_module\nlet x = 1", entry_path=None, roots=roots
        )
        rt = PipelineDriver()
        discovery = rt.discover_params(prepared)
        assert len(discovery.diagnostics) >= 1


# ---------------------------------------------------------------------------
# coverage gap tests for defensive/exceptional paths
# ---------------------------------------------------------------------------


class TestPreparedProgramDefensivePaths:
    """Edge-case coverage for PreparedProgram properties and prepare_program error paths."""

    def test_prepare_single_program_agl_error_retains_related_notes(self) -> None:
        """The module parse path uses AglError.to_diagnostic()."""
        from unittest.mock import patch

        related = SourceSpan(2, 1, 2, 2, 2, 3)
        error = AglError("parse failed", related=(("constraint", related),))
        with patch("agm.agl.modules.loader.parse_program_seeded", side_effect=error):
            prepared = prepare_inline_command("let x = 1")

        assert prepared.diagnostics[0].related[0].message == "constraint"

    def test_prepare_single_scope_agl_error_retains_related_notes(self) -> None:
        """The module scope path uses AglError.to_diagnostic()."""
        from unittest.mock import patch

        related = SourceSpan(2, 1, 2, 2, 2, 3)
        error = AglError("scope failed", related=(("constraint", related),))
        with patch("agm.agl.scope.program.resolve_program", side_effect=error):
            prepared = prepare_inline_command("let x = 1")

        assert prepared.diagnostics[0].related[0].message == "constraint"

    def test_prepare_program_agl_error_retains_related_notes(self, tmp_path: pathlib.Path) -> None:
        """The graph loader path uses AglError.to_diagnostic()."""
        from unittest.mock import patch

        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset({tmp_path.resolve(), _STDLIB_ROOT}))
        related = SourceSpan(2, 1, 2, 2, 2, 3)
        error = AglError("load failed", related=(("constraint", related),))
        with patch("agm.agl.modules.loader.build_repl_graph", side_effect=error):
            prepared = prepare_inline_command("let x = 1", entry_path=None, roots=roots)

        assert prepared.diagnostics[0].related[0].message == "constraint"

    def test_prepare_program_scope_agl_error_retains_related_notes(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The graph scope path uses AglError.to_diagnostic()."""
        from unittest.mock import patch

        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset({tmp_path.resolve(), _STDLIB_ROOT}))
        related = SourceSpan(2, 1, 2, 2, 2, 3)
        error = AglError("scope failed", related=(("constraint", related),))
        with patch("agm.agl.scope.program.resolve_program", side_effect=error):
            prepared = prepare_inline_command("let x = 1", entry_path=None, roots=roots)

        assert prepared.diagnostics[0].related[0].message == "constraint"

    def test_prepare_program_generic_exception_during_load(self, tmp_path: pathlib.Path) -> None:
        """A non-AglError exception during graph loading is captured as a diagnostic."""
        from unittest.mock import patch

        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset({tmp_path.resolve(), _STDLIB_ROOT}))
        with patch("agm.agl.modules.loader.build_repl_graph", side_effect=RuntimeError("boom")):
            prepared = prepare_inline_command("let x = 1\nx", entry_path=None, roots=roots)
        assert len(prepared.diagnostics) >= 1
        assert "boom" in prepared.diagnostics[0].message

    def test_prepare_program_generic_exception_during_resolve(self, tmp_path: pathlib.Path) -> None:
        """A non-AglScopeError exception during resolve_program is captured."""
        from unittest.mock import patch

        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset({tmp_path.resolve(), _STDLIB_ROOT}))
        with patch(
            "agm.agl.scope.program.resolve_program",
            side_effect=RuntimeError("resolve fail"),
        ):
            prepared = prepare_inline_command("let x = 1\nx", entry_path=None, roots=roots)
        assert len(prepared.diagnostics) >= 1
        assert "resolve fail" in prepared.diagnostics[0].message


class TestDiscoverParamsDefensivePaths:
    """Edge-case coverage for discover_params and discover_params."""

    def test_discover_params_missing_entry_in_checked(self, tmp_path: pathlib.Path) -> None:
        """discover_params returns diagnostic when checked graph has no entry module."""
        from unittest.mock import MagicMock, patch

        from agm.agl.modules.roots import RootSet

        roots = RootSet(roots=frozenset({_STDLIB_ROOT}))
        prepared = prepare_inline_command("let x = 1\nx", entry_path=None, roots=roots)
        rt = PipelineDriver()
        # Patch check_program to return a graph with no ENTRY_ID.
        fake_checked = MagicMock()
        fake_checked.modules = {}

        with patch("agm.agl.typecheck.program.check_program", return_value=fake_checked):
            discovery = rt.discover_params(prepared)
        assert len(discovery.diagnostics) >= 1
        assert "Entry module not found" in discovery.diagnostics[0].message

    def test_discover_params_typecheck_failure(self, tmp_path: pathlib.Path) -> None:
        """discover_params returns diagnostics when typecheck fails."""
        from unittest.mock import patch

        from agm.agl.typecheck import AglTypeError

        prepared = prepare_inline_command('def f(x: int) -> text = "bad"\nlet r = f(1)\nprint r')
        rt = PipelineDriver()
        with patch(
            "agm.agl.typecheck.program.check_program", side_effect=AglTypeError("type error")
        ):
            discovery = rt.discover_params(prepared)
        assert discovery.checked is None
        assert len(discovery.diagnostics) >= 1

    def test_discover_params_typecheck_agl_error_retains_related_notes(self) -> None:
        """The module typecheck boundary uses AglError.to_diagnostic()."""
        from unittest.mock import patch

        prepared = prepare_inline_command("let x = 1")
        related = SourceSpan(2, 1, 2, 2, 2, 3)
        error = AglError("type failed", related=(("constraint", related),))
        with patch("agm.agl.typecheck.program.check_program", side_effect=error):
            discovery = PipelineDriver().discover_params(prepared)

        assert discovery.diagnostics[0].related[0].message == "constraint"

    def test_run_typecheck_program_agl_error_retains_related_notes(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The graph typecheck boundary uses AglError.to_diagnostic()."""
        from unittest.mock import MagicMock, patch

        from agm.agl.modules.roots import RootSet
        from agm.agl.pipeline import PreparedProgram

        roots = RootSet(roots=frozenset({_STDLIB_ROOT}))
        prepared = PreparedProgram(
            source="let x = 1",
            entry_path=None,
            roots=roots,
            resolved=MagicMock(),
            diagnostics=(),
            warnings=(),
        )
        related = SourceSpan(2, 1, 2, 2, 2, 3)
        error = AglError("type failed", related=(("constraint", related),))
        with patch("agm.agl.typecheck.program.check_program", side_effect=error):
            discovery = PipelineDriver().discover_params(prepared)

        assert discovery.diagnostics[0].related[0].message == "constraint"

    def test_package_entry_param_qualifier_uses_the_module_route(
        self, tmp_path: pathlib.Path
    ) -> None:
        import semver

        from agm.agl.modules.roots import RootSet
        from agm.agl.pipeline import PreparedProgram, _entry_param_module_qualifier
        from agm.packages.manifest import PackageManifest
        from agm.packages.model import PackageInfo

        package_root = tmp_path / "package"
        entry_path = package_root / "tools" / "review.agl"
        entry_path.parent.mkdir(parents=True)
        entry_path.touch()
        package = PackageInfo(package_root, PackageManifest("tools", semver.Version.parse("1.0.0")))
        other_package = PackageInfo(
            tmp_path / "other", PackageManifest("other", semver.Version.parse("1.0.0"))
        )
        prepared = PreparedProgram(
            source="",
            entry_path=entry_path,
            roots=RootSet(
                frozenset({package.module_root, other_package.module_root}),
                packages=(other_package, package),
            ),
            resolved=None,
            diagnostics=(),
            warnings=(),
        )

        assert _entry_param_module_qualifier(prepared) == "tools/review"

    def test_run_typecheck_program_generic_exception_captured(self, tmp_path: pathlib.Path) -> None:
        """_run_typecheck_program captures generic exceptions as diagnostics."""
        from unittest.mock import MagicMock, patch

        from agm.agl.modules.roots import RootSet
        from agm.agl.pipeline import PreparedProgram

        roots = RootSet(roots=frozenset({_STDLIB_ROOT}))
        # Build a PreparedProgram with a fake resolved so we reach check_program.
        fake_rg = MagicMock()
        fake_rg.warnings = ()
        fake_rg.modules = {}

        pg = PreparedProgram(
            source="let x = 1",
            entry_path=None,
            roots=roots,
            resolved=fake_rg,
            diagnostics=(),
            warnings=(),
        )
        rt = PipelineDriver()
        with patch(
            "agm.agl.typecheck.program.check_program",
            side_effect=RuntimeError("graph type crash"),
        ):
            # discover_params will call _run_typecheck_program internally.
            discovery = rt.discover_params(pg)
        assert len(discovery.diagnostics) >= 1
        assert "graph type crash" in discovery.diagnostics[0].message


class TestRunPreparedDefensivePaths:
    """Edge-case coverage for run_prepared (module) and run_prepared."""

    def test_run_prepared_with_precomputed_compiled(self) -> None:
        """run_prepared skips static passes when a compiled artifact is passed."""
        from agm.agl.matchcompile import MatchCompiledProgram, compile_program_matches
        from agm.agl.typecheck.program import check_program

        prepared = prepare_inline_command("let x = 1\nx")
        assert prepared.resolved is not None
        rt = PipelineDriver()
        env = rt.host_environment()
        precomputed = check_program(prepared.resolved, env.capabilities)
        match_result = compile_program_matches(precomputed)
        assert isinstance(match_result.compiled, MatchCompiledProgram)
        result = rt.run_prepared(prepared, compiled=match_result.compiled)
        assert result.ok is True

    def test_run_prepared_typecheck_failure(self) -> None:
        """run_prepared exits early when typecheck fails (checked is None)."""
        from unittest.mock import patch

        from agm.agl.typecheck import AglTypeError

        prepared = prepare_inline_command("let x = 1\nx")
        rt = PipelineDriver()
        with patch("agm.agl.typecheck.program.check_program", side_effect=AglTypeError("tc fail")):
            result = rt.run_prepared(prepared)
        assert result.ok is False
        assert any("tc fail" in d.message for d in result.diagnostics)

    def test_run_prepared_custom_contract_error(self) -> None:
        """run_prepared exits early when custom pre-lower materialization fails."""
        from unittest.mock import patch

        from agm.agl.modules.roots import RootSet
        from agm.agl.runtime.codec import TextCodec

        class BadCodec(TextCodec):
            @property
            def name(self) -> str:
                return "bad"

        roots = RootSet(roots=frozenset({_STDLIB_ROOT}))
        prepared = prepare_inline_command(
            'let r = ask("hi", format = "bad")\nr', entry_path=None, roots=roots
        )
        rt = PipelineDriver(agent_dispatcher=lambda req: "ok")
        rt.register_codec(BadCodec())
        with patch("agm.agl.runtime.contract.materialize_contract", side_effect=ValueError("bad")):
            result = rt.run_prepared(prepared)
        assert result.ok is False
        assert any("Contract error" in d.message for d in result.diagnostics)

    def test_run_prepared_with_precompiled_decodes_param_from_ir(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A precompiled graph executes supplied params through IR metadata."""
        from agm.agl.matchcompile import MatchCompiledProgram, compile_program_matches
        from agm.agl.modules.roots import RootSet
        from agm.agl.typecheck.program import check_program as real_check_program

        roots = RootSet(roots=frozenset({_STDLIB_ROOT}))
        prepared = prepare_inline_command("param n: int\nprint n", entry_path=None, roots=roots)
        rt = PipelineDriver()
        env = rt.host_environment()
        assert prepared.resolved is not None
        checked = real_check_program(prepared.resolved, env.capabilities)
        match_result = compile_program_matches(checked)
        assert isinstance(match_result.compiled, MatchCompiledProgram)
        result = rt.run_prepared(prepared, param_values={"n": 7}, compiled=match_result.compiled)
        assert result.ok


class TestScopedAgentValues:
    def test_scoped_agent_values_keep_distinct_dispatches(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = """\
scope A
let bot = AgentCommand("a-bot")
end A
scope B
let bot = AgentCommand("b-bot")
end B
print(ask("first", agent = A::bot))
print(ask("second", agent = B::bot))
"""

        def dispatch(request: AgentRequest) -> str:
            command = request.agent.fields["command"]
            assert isinstance(command, TextValue)
            return {"a-bot": "from A", "b-bot": "from B"}[command.value]

        runtime = PipelineDriver(agent_dispatcher=dispatch)

        result = run_inline_command(runtime, source)

        assert result.ok
        assert capsys.readouterr().out == "from A\nfrom B\n"

    def test_scoped_agent_values_publish_distinct_full_path_names(self) -> None:
        from agm.agl.semantics.values import EnumValue

        source = """\
scope A
let bot = AgentCommand("a-bot")
end A
scope B
let bot = AgentCommand("b-bot")
end B
"""
        result = run_inline_command(PipelineDriver(), source)

        assert result.ok, result.diagnostics
        assert isinstance(result.bindings["A::bot"], EnumValue)
        assert isinstance(result.bindings["B::bot"], EnumValue)
        assert result.bindings["A::bot"] != result.bindings["B::bot"]

    def test_scoped_agent_value_does_not_mask_a_same_named_root_binding(self) -> None:
        from agm.agl.semantics.values import EnumValue, IntValue

        source = """\
let bot = 42
scope A
let bot = AgentCommand("a-bot")
end A
print(ask("hi", agent = A::bot))
"""
        runtime = PipelineDriver(agent_dispatcher=lambda _request: "hi there")

        result = run_inline_command(runtime, source)

        assert result.ok, result.diagnostics
        assert result.bindings["bot"] == IntValue(42)
        assert isinstance(result.bindings["A::bot"], EnumValue)


class TestScopedBindingPublicName:
    """A scoped ``let``/``var``'s public name is its full path spelling.

    ``RunResult.bindings`` is keyed by ``SymbolDescriptor.public_name``, so a
    root and a scoped binding sharing a bare name must not collide there.
    """

    def test_scoped_let_and_var_publish_full_path_names(self) -> None:
        from agm.agl.semantics.values import IntValue

        source = """\
let x = 1
var y = 2
scope A
let x = 10
var y = 20
end A
"""
        result = run_inline_command(PipelineDriver(), source)

        assert result.ok, result.diagnostics
        assert result.bindings["x"] == IntValue(1)
        assert result.bindings["y"] == IntValue(2)
        assert result.bindings["A::x"] == IntValue(10)
        assert result.bindings["A::y"] == IntValue(20)

    def test_shorthand_and_region_spellings_publish_the_same_name(self) -> None:
        from agm.agl.semantics.values import IntValue

        region_source = """\
scope A
let x = 1
end A
"""
        shorthand_source = "let A::x = 1"

        region_result = run_inline_command(PipelineDriver(), region_source)
        shorthand_result = PipelineDriver().run(shorthand_source)

        assert region_result.ok and shorthand_result.ok
        assert region_result.bindings["A::x"] == IntValue(1)
        assert shorthand_result.bindings["A::x"] == IntValue(1)

    def test_destructured_binder_in_a_region_publishes_its_full_path_name(self) -> None:
        from agm.agl.semantics.values import IntValue

        source = """\
record Point
  x: int
  y: int
scope A
let Point(x, y) = Point(x = 1, y = 2)
end A
"""
        result = run_inline_command(PipelineDriver(), source)

        assert result.ok, result.diagnostics
        assert result.bindings["A::x"] == IntValue(1)
        assert result.bindings["A::y"] == IntValue(2)
