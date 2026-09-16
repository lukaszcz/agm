"""Unit tests for the REPL renderer and meta-command dispatcher.

These are pure-data modules (no terminal), so they are tested directly:

- :func:`agm.agl.repl.render.render_entry_result` for each entry kind, warnings,
  pre-execution diagnostics, and runtime-error rendering;
- :func:`agm.agl.repl.meta.dispatch_meta` for ``:help`` / ``:quit`` / ``:exit`` /
  unknown commands.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from agm.agl.diagnostics import Diagnostic, RelatedDiagnostic
from agm.agl.ir.program import ValueDescriptors
from agm.agl.pipeline import RunError
from agm.agl.repl import meta as meta_mod
from agm.agl.repl import render as render_mod
from agm.agl.repl.entry import EntryKind, EntryResult
from agm.agl.repl.session import ReplSession
from agm.agl.runtime.request import AgentRequest, AgentResponse
from agm.agl.semantics.types import IntType, TextType, Type
from agm.agl.semantics.values import IntValue, TextValue, Value

# A descriptor view with no nominal/function entries: enough for every test
# value here, none of which is a record/enum/exception or closure.
_EMPTY_DESCRIPTORS = ValueDescriptors(nominals={}, functions={})


def _open_session(**kwargs: object) -> ReplSession:
    """Build a session and open it, exactly as a REPL host does.

    ``agm.commands.repl`` opens a session before accepting an entry, which
    loads and type-checks the initial library image once per session instead
    of leaving every entry to re-check the standard library.
    """
    session = ReplSession(**kwargs)
    session.open()
    return session


class _CountingAgent:
    """Fake ``AgentFn`` returning scripted replies and counting invocations."""

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies) or ["ok"]
        self._next = 0
        self.calls = 0

    def __call__(self, request: AgentRequest) -> AgentResponse:
        del request
        self.calls += 1
        reply = self._replies[min(self._next, len(self._replies) - 1)]
        self._next += 1
        return AgentResponse(content=reply)


def _result(
    *,
    kind: EntryKind = "statement",
    name: str | None = None,
    value: Value | None = None,
    value_type: Type | None = None,
    diagnostics: list[Diagnostic] | None = None,
    warnings: list[Diagnostic] | None = None,
    error: RunError | None = None,
    ok: bool = True,
    installed: tuple[str, ...] = (),
    quote_strings: bool = True,
    descriptors: ValueDescriptors | None = None,
) -> EntryResult:
    """Build an ``EntryResult`` with sensible defaults for one test axis.

    *descriptors* defaults to an empty view whenever *value* is set (matching
    the invariant ``render_entry_result`` relies on: descriptors are present
    whenever a value is), and to ``None`` otherwise.
    """
    return EntryResult(
        kind=kind,
        name=name,
        value=value,
        value_type=value_type,
        diagnostics=diagnostics if diagnostics is not None else [],
        warnings=warnings if warnings is not None else [],
        error=error,
        ok=ok,
        installed=installed,
        quote_strings=quote_strings,
        descriptors=descriptors
        if descriptors is not None
        else (_EMPTY_DESCRIPTORS if value is not None else None),
    )


# ---------------------------------------------------------------------------
# render_entry_result
# ---------------------------------------------------------------------------


class TestRenderEntryResult:
    def test_expression_echo(self) -> None:
        result = _result(
            kind="expression", value=IntValue(Decimal(3)), value_type=IntType(), ok=True
        )
        assert render_mod.render_entry_result(result, echo=True) == "3"

    def test_expression_echo_quotes_text(self) -> None:
        # Strings are shown quoted in the REPL echo (interpolation is unaffected).
        result = _result(kind="expression", value=TextValue("aaa"), value_type=TextType(), ok=True)
        assert render_mod.render_entry_result(result, echo=True) == '"aaa"'

    def test_standalone_ask_echo_does_not_quote_text(self) -> None:
        result = _result(
            kind="expression",
            value=TextValue("aaa"),
            value_type=TextType(),
            ok=True,
            quote_strings=False,
        )
        assert render_mod.render_entry_result(result, echo=True) == "aaa"

    # The REPL binding echo is user-visible output with a stable format; these exact-string
    # assertions intentionally pin that contract (cf. the no-exact-error-message rule, which
    # applies to diagnostics, not user-facing REPL output).
    def test_binding_echo_quotes_text(self) -> None:
        result = _result(
            kind="binding",
            name="g",
            value=TextValue("hi"),
            value_type=TextType(),
            ok=True,
        )
        rendered = render_mod.render_entry_result(result, echo=True)
        assert rendered == 'g : text = "hi"'

    def test_binding_echo(self) -> None:
        result = _result(
            kind="binding",
            name="x",
            value=IntValue(Decimal(5)),
            value_type=IntType(),
            ok=True,
        )
        rendered = render_mod.render_entry_result(result, echo=True)
        assert rendered == "x : int = 5"

    def test_expression_echo_pretty_prints_structured_values(self) -> None:
        from agm.agl.ir.ids import NominalId
        from agm.agl.ir.program import NominalDescriptor, NominalKind
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.semantics.values import ArrayValue, IntValue, RecordValue

        nominal = NominalId(1)
        value = RecordValue(
            nominal=nominal,
            fields={"items": ArrayValue([IntValue(1), IntValue(2)])},
        )
        descriptors = ValueDescriptors(
            nominals={
                nominal: NominalDescriptor(
                    nominal=nominal,
                    module_id=ENTRY_ID,
                    scope_path=(),
                    declared_name="Box",
                    kind=NominalKind.RECORD,
                    fields=("items",),
                )
            },
            functions={},
        )
        result = _result(
            kind="expression",
            value=value,
            value_type=TextType(),
            descriptors=descriptors,
            ok=True,
        )

        rendered = render_mod.render_entry_result(result, echo=True)
        assert rendered == "Box(\n  items = [\n    1,\n    2\n  ]\n)"

    def test_declaration_echo(self) -> None:
        result = _result(kind="declaration", name="R", ok=True)
        assert render_mod.render_entry_result(result, echo=True) == "R declared"

    def test_statement_echo_is_none(self) -> None:
        result = _result(kind="statement", ok=True)
        assert render_mod.render_entry_result(result, echo=True) is None

    def test_unit_expression_echo_is_none_by_default(self) -> None:
        from agm.agl.semantics.types import UnitType
        from agm.agl.semantics.values import UNIT_VALUE

        result = _result(
            kind="expression",
            value=UNIT_VALUE,
            value_type=UnitType(),
            ok=True,
        )
        assert render_mod.render_entry_result(result, echo=True) is None

    def test_unit_binding_echo_is_none_by_default(self) -> None:
        from agm.agl.semantics.types import UnitType
        from agm.agl.semantics.values import UNIT_VALUE

        result = _result(
            kind="binding",
            name="x",
            value=UNIT_VALUE,
            value_type=UnitType(),
            ok=True,
        )
        assert render_mod.render_entry_result(result, echo=True) is None

    def test_unit_expression_echoes_when_echo_unit_on(self) -> None:
        from agm.agl.semantics.types import UnitType
        from agm.agl.semantics.values import UNIT_VALUE

        result = _result(kind="expression", value=UNIT_VALUE, value_type=UnitType(), ok=True)
        assert render_mod.render_entry_result(result, echo=True, echo_unit=True) == "()"

    def test_unit_binding_echoes_when_echo_unit_on(self) -> None:
        from agm.agl.semantics.types import UnitType
        from agm.agl.semantics.values import UNIT_VALUE

        result = _result(kind="binding", name="x", value=UNIT_VALUE, value_type=UnitType(), ok=True)
        rendered = render_mod.render_entry_result(result, echo=True, echo_unit=True)
        assert rendered == "x : unit = ()"

    def test_unit_entry_still_suppressed_when_echo_is_off(self) -> None:
        from agm.agl.semantics.types import UnitType
        from agm.agl.semantics.values import UNIT_VALUE

        result = _result(kind="expression", value=UNIT_VALUE, value_type=UnitType(), ok=True)
        assert render_mod.render_entry_result(result, echo=False, echo_unit=True) is None

    def test_dry_run_unit_expression_echoes_type_when_echo_unit_on(self) -> None:
        from agm.agl.semantics.types import UnitType

        result = _result(kind="expression", value_type=UnitType(), ok=True)
        rendered = render_mod.render_entry_result(
            result, echo=True, echo_unit=True, check_only=True
        )
        assert rendered == ": unit"

    def test_dry_run_unit_expression_echoes_nothing_by_default(self) -> None:
        from agm.agl.semantics.types import UnitType

        result = _result(kind="expression", value_type=UnitType(), ok=True)
        rendered = render_mod.render_entry_result(result, echo=True, check_only=True)
        assert rendered is None

    def test_check_only_expression_shows_type(self) -> None:
        # In dry-run there is no value; the echo shows the inferred type.
        result = _result(kind="expression", value_type=IntType(), ok=True)
        rendered = render_mod.render_entry_result(result, echo=True, check_only=True)
        assert rendered == ": int"

    def test_check_only_binding_shows_name_and_type(self) -> None:
        result = _result(kind="binding", name="x", value_type=IntType(), ok=True)
        rendered = render_mod.render_entry_result(result, echo=True, check_only=True)
        assert rendered == "x : int"

    def test_check_only_declaration_confirms_name(self) -> None:
        result = _result(kind="declaration", name="R", ok=True)
        rendered = render_mod.render_entry_result(result, echo=True, check_only=True)
        assert rendered == "R declared"

    def test_check_only_statement_is_none(self) -> None:
        result = _result(kind="statement", ok=True)
        assert render_mod.render_entry_result(result, echo=True, check_only=True) is None

    def test_check_only_echo_off_suppresses(self) -> None:
        result = _result(kind="expression", value_type=IntType(), ok=True)
        assert render_mod.render_entry_result(result, echo=False, check_only=True) is None

    def test_echo_off_suppresses_success(self) -> None:
        result = _result(kind="expression", value=TextValue("hi"), value_type=TextType(), ok=True)
        assert render_mod.render_entry_result(result, echo=False) is None

    def test_warnings_are_always_rendered(self) -> None:
        result = _result(
            kind="statement",
            ok=True,
            warnings=[
                Diagnostic(
                    message="watch out",
                    line=2,
                    column=5,
                    end_line=2,
                    end_column=10,
                    severity="warning",
                )
            ],
        )
        rendered = render_mod.render_entry_result(result, echo=False)
        assert rendered == "2:5-9: warning: watch out"

    def test_pre_execution_diagnostics(self) -> None:
        result = _result(
            ok=False,
            diagnostics=[Diagnostic(message="boom", line=1, column=4)],
        )
        assert render_mod.render_entry_result(result, echo=True) == "1:4: error: boom"

    def test_pre_execution_related_diagnostics_are_rendered(self) -> None:
        result = _result(
            ok=False,
            diagnostics=[
                Diagnostic(
                    message="boom",
                    line=1,
                    column=4,
                    related=(RelatedDiagnostic(message="earlier", line=2, column=3),),
                )
            ],
        )
        assert render_mod.render_entry_result(result, echo=True) == (
            "1:4: error: boom\n  2:3: note: earlier"
        )

    def test_runtime_failure_reports_partially_installed_names(self) -> None:
        result = _result(
            ok=False,
            diagnostics=[Diagnostic(message="boom", line=1)],
            installed=("before", "Box"),
        )
        assert render_mod.render_entry_result(result, echo=True) == (
            "1: error: boom\nInstalled before failure: before, Box"
        )

    def test_runtime_error_with_location(self) -> None:
        result = _result(
            ok=False,
            error=RunError(
                type_name="MyError",
                fields={"message": "bad"},
                line=4,
                col=7,
            ),
        )
        rendered = render_mod.render_entry_result(result, echo=True)
        assert rendered == "AgL exception: MyError: bad: at line 4, col 7"

    def test_runtime_error_line_only(self) -> None:
        result = _result(
            ok=False,
            error=RunError(type_name="MyError", fields={}, line=4, col=None),
        )
        rendered = render_mod.render_entry_result(result, echo=True)
        assert rendered == "AgL exception: MyError: at line 4"

    def test_runtime_error_no_location(self) -> None:
        result = _result(
            ok=False,
            error=RunError(type_name="MyError", fields={}, line=None, col=None),
        )
        rendered = render_mod.render_entry_result(result, echo=True)
        assert rendered == "AgL exception: MyError"

    def test_warning_then_error(self) -> None:
        result = _result(
            ok=False,
            warnings=[Diagnostic(message="w", line=1, column=1, severity="warning")],
            diagnostics=[Diagnostic(message="e", line=2, column=3)],
        )
        rendered = render_mod.render_entry_result(result, echo=True)
        assert rendered == "1:1: warning: w\n2:3: error: e"


# ---------------------------------------------------------------------------
# dispatch_meta
# ---------------------------------------------------------------------------


def _ctx() -> meta_mod.MetaContext:
    return meta_mod.MetaContext(session=_open_session())


class TestDispatchMeta:
    def test_help_lists_commands(self) -> None:
        outcome = meta_mod.dispatch_meta(":help", _ctx())
        assert outcome.text is not None
        assert ":help" in outcome.text
        assert ":quit" in outcome.text
        assert outcome.quit is False

    def test_quit(self) -> None:
        ctx = _ctx()
        outcome = meta_mod.dispatch_meta(":quit", ctx)
        assert outcome.quit is True
        assert ctx.quit is True

    def test_exit_alias(self) -> None:
        outcome = meta_mod.dispatch_meta(":exit", _ctx())
        assert outcome.quit is True

    def test_unknown_command(self) -> None:
        outcome = meta_mod.dispatch_meta(":nope", _ctx())
        assert outcome.text is not None
        assert "Unknown command ':nope'" in outcome.text
        assert outcome.quit is False

    def test_params_is_an_unknown_command(self) -> None:
        # The REPL has no entry function, so ":params" is not a meta command.
        outcome = meta_mod.dispatch_meta(":params", _ctx())
        assert outcome.text is not None
        assert "Unknown command ':params'" in outcome.text
        assert outcome.quit is False

    def test_argument_is_ignored_for_known_command(self) -> None:
        # Trailing text after the command word is passed to the handler; :help
        # ignores it and still prints the list.
        outcome = meta_mod.dispatch_meta(":help everything", _ctx())
        assert outcome.text is not None
        assert ":quit" in outcome.text

    def test_meta_command_names_includes_aliases(self) -> None:
        names = meta_mod.meta_command_names()
        assert ":help" in names
        assert ":quit" in names
        assert ":exit" in names

    def test_dispatch_tab_separated_command(self) -> None:
        # Issue #3: tab (or other whitespace) between command word and arg must
        # dispatch correctly, not produce "Unknown command".
        outcome = meta_mod.dispatch_meta(":type\tx + 1", _ctx())
        # ":type" with no valid binding → some error message, but NOT "Unknown command"
        assert outcome.text is not None
        assert "Unknown command" not in outcome.text

    def test_dispatch_tab_command_no_arg(self) -> None:
        # A tab immediately after a known command name with no trailing arg.
        outcome = meta_mod.dispatch_meta(":help\t", _ctx())
        assert outcome.text is not None
        assert ":quit" in outcome.text

    def test_command_index_cold_cache(self) -> None:
        # _command_index() must rebuild if its cache is None (cold-start path).
        original = meta_mod._command_index_cache
        meta_mod._command_index_cache = None
        try:
            idx = meta_mod._command_index()
            assert ":help" in idx or "help" in idx
        finally:
            meta_mod._command_index_cache = original

    def test_command_index_lazy_build_on_cold_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Verify the lazy-build branch of _command_index() is reachable: when both
        # caches are forced to None the functions rebuild them on the next call.
        monkeypatch.setattr(meta_mod, "_command_index_cache", None)
        monkeypatch.setattr(meta_mod, "_command_names_cache", None)
        # dispatch_meta → _command_index() triggers the lazy build of the index.
        outcome = meta_mod.dispatch_meta(":help", _ctx())
        assert outcome.text is not None
        assert ":quit" in outcome.text
        # meta_command_names() rebuilds the names cache when it is cold.
        monkeypatch.setattr(meta_mod, "_command_names_cache", None)
        names = meta_mod.meta_command_names()
        assert ":help" in names


# ---------------------------------------------------------------------------
# :set only controls REPL options
# ---------------------------------------------------------------------------


class TestSetOptions:
    def test_set_non_option_gives_usage(self) -> None:
        s = _open_session()
        outcome = meta_mod.dispatch_meta(":set count=5", _session_ctx(s))
        assert outcome.text is not None
        assert "usage" in outcome.text.lower()

    @pytest.mark.parametrize(
        "arg",
        ("foo", "=5", "foo=", "", "echo", "echo on off", "echo-unit", "echo-unit maybe"),
        ids=(
            "missing-equals",
            "empty-name",
            "empty-value",
            "no-argument",
            "echo-without-state",
            "echo-with-extra-words",
            "echo-unit-without-state",
            "echo-unit-bad-state",
        ),
    )
    def test_set_non_echo_forms_give_usage(self, arg: str) -> None:
        outcome = meta_mod.dispatch_meta(f":set {arg}".rstrip(), _session_ctx())
        assert "usage" in (outcome.text or "").lower()


# ---------------------------------------------------------------------------
# render helpers (single-sourced binding/value formatting)
# ---------------------------------------------------------------------------


class TestRenderHelpers:
    def test_format_typed_value(self) -> None:
        line = render_mod.format_typed_value(
            "x", IntType(), IntValue(Decimal(5)), _EMPTY_DESCRIPTORS
        )
        assert line == "x : int = 5"

    def test_binding_echo_matches_format_helper(self) -> None:
        # The entry-echo path and the shared helper must produce the same line.
        value = TextValue("hi")
        result = _result(kind="binding", name="g", value=value, value_type=TextType())
        echoed = render_mod.render_entry_result(result, echo=True)
        assert echoed == render_mod.format_typed_value("g", TextType(), value, _EMPTY_DESCRIPTORS)

    def test_format_typed_value_without_name(self) -> None:
        line = render_mod.format_typed_value(
            None, IntType(), IntValue(Decimal(5)), _EMPTY_DESCRIPTORS
        )
        assert line == ": int = 5"

    def test_unnamed_binding_echo_matches_format_helper(self) -> None:
        # A destructuring let has no single public name (``result.name is
        # None``); its echo must go through the SAME shared formatter as a
        # named binding, just with no name, so the two never drift.
        value = TextValue("hi")
        result = _result(kind="binding", name=None, value=value, value_type=TextType())
        echoed = render_mod.render_entry_result(result, echo=True)
        assert echoed == render_mod.format_typed_value(None, TextType(), value, _EMPTY_DESCRIPTORS)


# ---------------------------------------------------------------------------
# Full meta-command set
# ---------------------------------------------------------------------------


def _session_ctx(session: ReplSession | None = None) -> meta_mod.MetaContext:
    return meta_mod.MetaContext(session=session if session is not None else _open_session())


class TestReset:
    def test_reset_clears_bindings(self) -> None:
        s = _open_session()
        s.eval_entry("let x = 1")
        assert s.bindings()
        outcome = meta_mod.dispatch_meta(":reset", _session_ctx(s))
        assert "reset" in (outcome.text or "").lower()
        assert s.bindings() == []


class TestType:
    def test_type_of_valid_expr(self) -> None:
        s = _open_session()
        s.eval_entry("let x = 5")
        outcome = meta_mod.dispatch_meta(":type x + 1", _session_ctx(s))
        assert outcome.text == "int"

    def test_type_of_record_expr_shows_fields(self) -> None:
        s = _open_session()
        s.eval_entry("record Point\n  x: int\n  y: text")
        s.eval_entry('let p = Point(x = 1, y = "north")')
        outcome = meta_mod.dispatch_meta(":type p", _session_ctx(s))
        assert outcome.text == "record Point\n  x: int\n  y: text"

    def test_empty_record_type_display_uses_empty_constructor_form(self) -> None:
        from agm.agl.modules.ids import ENTRY_ID
        from agm.agl.repl.type_display import format_type_for_repl
        from agm.agl.semantics.type_table import TypeDef, TypeTable
        from agm.agl.semantics.types import RecordType

        table = TypeTable()
        table.register(TypeDef(kind="record", name="Empty", module_id=ENTRY_ID, decl_node_id=1))
        assert format_type_for_repl(RecordType(name="Empty", decl_id=1), table) == "record Empty()"

    def test_type_empty_arg_gives_usage(self) -> None:
        outcome = meta_mod.dispatch_meta(":type", _session_ctx())
        assert "usage" in (outcome.text or "").lower()

    def test_type_unknown_name_clean_error(self) -> None:
        outcome = meta_mod.dispatch_meta(":type nope", _session_ctx())
        assert outcome.text is not None
        assert outcome.text.startswith("1:1")
        assert "error:" in outcome.text
        assert outcome.quit is False  # never crashed the loop

    def test_type_match_error_preserves_source_location(self) -> None:
        outcome = meta_mod.dispatch_meta(":type case true of | true => 1", _session_ctx())
        assert outcome.text is not None
        assert outcome.text.startswith("1:1")
        assert "Non-exhaustive" in outcome.text

    def test_type_bad_syntax_clean_error(self) -> None:
        outcome = meta_mod.dispatch_meta(":type 1 +", _session_ctx())
        assert outcome.text is not None

    def test_type_non_expression_clean_error(self) -> None:
        # A binding is not a single expression — type_of raises AglError, caught.
        outcome = meta_mod.dispatch_meta(":type let y = 1", _session_ctx())
        assert outcome.text is not None
        assert "expression" in outcome.text.lower()


class TestInfo:
    def test_info_reports_a_binding_type_value_and_location(self) -> None:
        session = _open_session()
        assert session.eval_entry("var count = 3").ok

        outcome = meta_mod.dispatch_meta(":info count", _session_ctx(session))

        assert outcome.text is not None
        assert "binding" in outcome.text.lower()
        assert "int" in outcome.text
        assert "3" in outcome.text
        assert "mutable" in outcome.text.lower()
        assert "<repl>:1:" in outcome.text

    def test_info_reports_a_function_signature_and_location(self) -> None:
        session = _open_session()
        assert session.eval_entry("def twice(value: int) -> int = value * 2").ok

        outcome = meta_mod.dispatch_meta(":info twice", _session_ctx(session))

        assert outcome.text is not None
        assert "function" in outcome.text.lower()
        assert "value: int" in outcome.text
        assert "-> int" in outcome.text
        assert "<repl>:1:" in outcome.text

    def test_info_reports_a_type_definition(self) -> None:
        session = _open_session()
        assert session.eval_entry("record Point\n  x: int\n  y: int").ok

        outcome = meta_mod.dispatch_meta(":info Point", _session_ctx(session))

        assert outcome.text is not None
        assert "type" in outcome.text.lower()
        assert "record Point" in outcome.text
        assert "x: int" in outcome.text

    def test_info_reports_alias_and_generic_type_definitions(self) -> None:
        session = _open_session()
        assert session.eval_entry("type Count = int").ok
        assert session.eval_entry("record Box[T]\n  value: T").ok

        alias = meta_mod.dispatch_meta(":info Count", _session_ctx(session))
        generic = meta_mod.dispatch_meta(":info Box", _session_ctx(session))

        assert alias.text is not None
        assert "alias" in alias.text.lower()
        assert "Count = int" in alias.text
        assert generic.text is not None
        assert "generic" in generic.text.lower()
        assert "record Box[T]" in generic.text

    def test_info_rejects_an_invalid_qualified_name(self) -> None:
        assert _open_session().info_of("Count::") is None

    def test_info_reports_the_static_type_of_an_imported_value(self) -> None:
        session = _open_session()
        assert session.eval_entry("import std/config").ok

        outcome = meta_mod.dispatch_meta(":info std/config::strict-json", _session_ctx(session))

        assert outcome.text is not None
        assert "value" in outcome.text.lower()
        assert "bool" in outcome.text

    def test_info_requires_one_known_identifier(self) -> None:
        assert "usage" in (meta_mod.dispatch_meta(":info", _session_ctx()).text or "").lower()
        assert (
            "unknown"
            in (meta_mod.dispatch_meta(":info missing", _session_ctx()).text or "").lower()
        )

    def test_info_formats_full_function_parameter_metadata(self) -> None:
        from agm.agl.repl.session import _format_repl_signature
        from agm.agl.semantics.types import BoolType, IntType, TextType
        from agm.agl.typecheck.env import FunctionSignature, ParamSpec
        from agm.agl.zones import ParamZone

        signature = FunctionSignature(
            params=(
                ParamSpec("first", IntType(), ParamZone.POSITIONAL_ONLY, False),
                ParamSpec("second", TextType(), ParamZone.STANDARD, True),
                ParamSpec("third", BoolType(), ParamZone.NAMED_ONLY, False),
            ),
            result=IntType(),
            type_params=("T",),
        )

        rendered = _format_repl_signature(signature)

        assert "[T]" in rendered
        assert "positional-only" in rendered
        assert "= …" in rendered
        assert "named-only" in rendered


class TestBindings:
    def test_bindings_empty(self) -> None:
        for name in (":bindings", ":env"):
            outcome = meta_mod.dispatch_meta(name, _session_ctx())
            assert outcome.text == "No bindings."

    def test_bindings_lists_with_types_and_values(self) -> None:
        s = _open_session()
        s.eval_entry("let x = 5")
        s.eval_entry('let g = "hi"')
        outcome = meta_mod.dispatch_meta(":bindings", _session_ctx(s))
        assert outcome.text is not None
        assert "x : int = 5" in outcome.text
        assert 'g : text = "hi"' in outcome.text

    def test_env_alias_same_as_bindings(self) -> None:
        s = _open_session()
        s.eval_entry("let x = 5")
        out_b = meta_mod.dispatch_meta(":bindings", _session_ctx(s)).text
        out_e = meta_mod.dispatch_meta(":env", _session_ctx(s)).text
        assert out_b == out_e


class TestSet:
    def test_set_non_echo_argument_reports_usage(self) -> None:
        outcome = meta_mod.dispatch_meta(":set count=42", _session_ctx())
        assert "usage" in (outcome.text or "").lower()

    def test_set_echo_off_then_on_toggles_ctx(self) -> None:
        ctx = _session_ctx()
        assert ctx.echo is True
        off = meta_mod.dispatch_meta(":set echo off", ctx)
        assert ctx.echo is False
        assert "off" in (off.text or "").lower()
        assert off.setting_change == ("echo", False)
        on = meta_mod.dispatch_meta(":set echo on", ctx)
        assert ctx.echo is True
        assert "on" in (on.text or "").lower()
        assert on.setting_change == ("echo", True)

    def test_set_echo_bad_state_gives_usage(self) -> None:
        ctx = _session_ctx()
        outcome = meta_mod.dispatch_meta(":set echo maybe", ctx)
        assert "usage" in (outcome.text or "").lower()
        assert ctx.echo is True  # unchanged
        assert outcome.setting_change is None

    def test_set_echo_already_on_still_reports_the_change_to_persist(self) -> None:
        # Already on; setting it to "on" again live-changes nothing, but the
        # explicit command still reports its target for persistence.
        ctx = _session_ctx()
        outcome = meta_mod.dispatch_meta(":set echo on", ctx)
        assert ctx.echo is True
        assert outcome.setting_change == ("echo", True)

    def test_set_echo_unit_off_then_on_toggles_ctx(self) -> None:
        ctx = _session_ctx()
        assert ctx.echo_unit is False
        on = meta_mod.dispatch_meta(":set echo-unit on", ctx)
        assert ctx.echo_unit is True
        assert "on" in (on.text or "").lower()
        assert on.setting_change == ("echo-unit", True)
        off = meta_mod.dispatch_meta(":set echo-unit off", ctx)
        assert ctx.echo_unit is False
        assert "off" in (off.text or "").lower()
        assert off.setting_change == ("echo-unit", False)

    def test_set_echo_unit_bad_state_gives_usage(self) -> None:
        ctx = _session_ctx()
        outcome = meta_mod.dispatch_meta(":set echo-unit maybe", ctx)
        assert "usage" in (outcome.text or "").lower()
        assert ctx.echo_unit is False  # unchanged
        assert outcome.setting_change is None


class TestLoad:
    def test_load_runs_file_into_session(self, tmp_path: Path) -> None:
        src = tmp_path / "prog.agl"
        src.write_text("let x = 7\n")
        s = _open_session()
        outcome = meta_mod.dispatch_meta(f":load {src}", _session_ctx(s))
        assert outcome.text is not None
        assert "x : int = 7" in outcome.text
        # The binding persisted into the session.
        assert any(n == "x" for n, _t, _v in s.bindings())

    def test_load_agent_call_fires_exactly_once(self, tmp_path: Path) -> None:
        src = tmp_path / "agent.agl"
        src.write_text('let r = ask """do it"""\n')
        agent = _CountingAgent("done")
        s = _open_session(agent_dispatcher=agent)
        meta_mod.dispatch_meta(f":load {src}", _session_ctx(s))
        assert agent.calls == 1

    def test_load_missing_file_clean_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.agl"
        outcome = meta_mod.dispatch_meta(f":load {missing}", _session_ctx())
        assert outcome.text is not None
        assert "cannot read" in outcome.text.lower()

    def test_load_empty_arg_usage(self) -> None:
        outcome = meta_mod.dispatch_meta(":load", _session_ctx())
        assert "usage" in (outcome.text or "").lower()

    def test_load_renders_each_statement(self, tmp_path: Path) -> None:
        # Multiple statements load incrementally; each statement's echo surfaces.
        src = tmp_path / "multi.agl"
        src.write_text("let a = 1\nlet b = 2\n")
        s = _open_session()
        outcome = meta_mod.dispatch_meta(f":load {src}", _session_ctx(s))
        assert outcome.text is not None
        assert "a : int = 1" in outcome.text
        assert "b : int = 2" in outcome.text

    def test_load_halts_at_first_error(self, tmp_path: Path) -> None:
        src = tmp_path / "halt.agl"
        src.write_text("let a = 1\nlet z: decimal = 1 / 0\nlet b = 99\n")
        s = _open_session()
        outcome = meta_mod.dispatch_meta(f":load {src}", _session_ctx(s))
        assert outcome.text is not None
        # The failing statement's error surfaced; the unreached one did not run.
        assert "exception" in outcome.text.lower()
        names = {n for n, _t, _v in s.bindings()}
        assert names == {"a"}

    def test_load_empty_file_benign_note(self, tmp_path: Path) -> None:
        src = tmp_path / "empty.agl"
        src.write_text("# only a comment\n")
        s = _open_session()
        outcome = meta_mod.dispatch_meta(f":load {src}", _session_ctx(s))
        assert outcome.text is not None
        assert "no statements" in outcome.text.lower()
        assert s.bindings() == []


class TestSave:
    def test_save_round_trips_source(self, tmp_path: Path) -> None:
        s = _open_session()
        s.eval_entry("let x = 1")
        s.eval_entry("let y = 2")
        out = tmp_path / "out.agl"
        outcome = meta_mod.dispatch_meta(f":save {out}", _session_ctx(s))
        assert str(out) in (outcome.text or "")
        assert out.read_text() == s.dump_source()
        # The saved source replays into a fresh session.
        s2 = _open_session()
        assert all(r.ok for r in s2.load_file(out))

    def test_save_load_round_trips_redefinition(self, tmp_path: Path) -> None:
        # A transcript containing a redefinition must round-trip through
        # :save -> :load (each statement loads as its own entry, so the second
        # `let x` shadows rather than being a duplicate-declaration error).
        s = _open_session()
        s.eval_entry("let x = 1")
        s.eval_entry("let x = 2")
        out = tmp_path / "redef.agl"
        meta_mod.dispatch_meta(f":save {out}", _session_ctx(s))

        s2 = _open_session()
        outcome = meta_mod.dispatch_meta(f":load {out}", _session_ctx(s2))
        # No error surfaced and x reloaded as the shadowed value 2.
        assert "line" not in (outcome.text or "")
        assert "x : int = 2" in (outcome.text or "")
        vals = {n: v for n, _t, v in s2.bindings()}
        assert isinstance(vals["x"], IntValue)
        assert vals["x"].value == 2

    def test_save_empty_arg_usage(self) -> None:
        outcome = meta_mod.dispatch_meta(":save", _session_ctx())
        assert "usage" in (outcome.text or "").lower()

    def test_save_unwritable_path_clean_error(self, tmp_path: Path) -> None:
        # A path whose parent directory does not exist cannot be written.
        bad = tmp_path / "missing_dir" / "out.agl"
        outcome = meta_mod.dispatch_meta(f":save {bad}", _session_ctx())
        assert outcome.text is not None
        assert "cannot write" in outcome.text.lower()


class TestTheme:
    def test_no_arg_reports_current_theme(self) -> None:
        ctx = _ctx()
        ctx.theme = "dark"
        outcome = meta_mod.dispatch_meta(":theme", ctx)
        assert outcome.text is not None
        assert "dark" in outcome.text

    def test_switch_to_light(self) -> None:
        ctx = _ctx()
        outcome = meta_mod.dispatch_meta(":theme light", ctx)
        assert ctx.theme == "light"
        assert outcome.text is not None
        assert "light" in outcome.text
        assert outcome.setting_change == ("theme", "light")

    def test_switch_to_dark(self) -> None:
        ctx = _ctx()
        ctx.theme = "light"
        meta_mod.dispatch_meta(":theme dark", ctx)
        assert ctx.theme == "dark"

    def test_switch_to_auto(self) -> None:
        ctx = _ctx()
        meta_mod.dispatch_meta(":theme auto", ctx)
        assert ctx.theme == "auto"

    def test_switch_to_same_theme_still_reports_the_change_to_persist(self) -> None:
        ctx = _ctx()
        ctx.theme = "dark"
        outcome = meta_mod.dispatch_meta(":theme dark", ctx)
        assert outcome.setting_change == ("theme", "dark")

    def test_unknown_theme_returns_error(self) -> None:
        ctx = _ctx()
        outcome = meta_mod.dispatch_meta(":theme neon", ctx)
        assert outcome.text is not None
        assert "Unknown theme" in outcome.text
        assert ctx.theme == "auto"  # unchanged
        assert outcome.setting_change is None

    def test_theme_in_help(self) -> None:
        outcome = meta_mod.dispatch_meta(":help", _ctx())
        assert outcome.text is not None
        assert ":theme" in outcome.text

    def test_theme_in_meta_command_names(self) -> None:
        assert ":theme" in meta_mod.meta_command_names()


# ---------------------------------------------------------------------------
# nominal value rendering (record/enum in AgL form and declaration order)
# ---------------------------------------------------------------------------


class TestNominalRenderingEcho:
    """Verify REPL echo renders records and enums in AgL form / declaration order."""

    def test_record_binding_keeps_declaration_order(self) -> None:
        s = _open_session()
        s.eval_entry("record Point\n  y: int\n  x: int")
        r = s.eval_entry("let p = Point(x = 1, y = 2)")

        assert r.ok
        assert render_mod.render_entry_result(r, echo=True) == (
            "p : Point = Point(\n  y = 2,\n  x = 1\n)"
        )
        outcome = meta_mod.dispatch_meta(":bindings", _session_ctx(s))
        assert outcome.text is not None
        assert "p : Point = Point(\n  y = 2,\n  x = 1\n)" in outcome.text

    def test_enum_binding_echoes_qualified_value(self) -> None:
        s = _open_session()
        s.eval_entry("enum Outcome\n  | Partial(left: int)\n  | Done")
        r = s.eval_entry("let o = Outcome::Partial(left = 7)")

        assert r.ok
        assert render_mod.render_entry_result(r, echo=True) == (
            "o : Outcome::Partial = Outcome::Partial(\n  left = 7\n)"
        )
        outcome = meta_mod.dispatch_meta(":bindings", _session_ctx(s))
        assert outcome.text is not None
        assert "Outcome::Partial(\n  left = 7\n)" in outcome.text

    def test_percent_interpolation_marker_in_text_binding_renders_in_bindings(self) -> None:
        s = _open_session()
        r = s.eval_entry(r'let t = "a\%{b}"')

        assert r.ok
        outcome = meta_mod.dispatch_meta(":bindings", _session_ctx(s))
        assert outcome.text is not None
        assert r"a\%{b}" in outcome.text

    def test_bindings_meta_renders_record_nominal(self) -> None:
        # :bindings must render a record binding in AgL form (not JSON).
        s = _open_session()
        s.eval_entry("record Point\n  y: int\n  x: int")
        s.eval_entry("let p = Point(x = 3, y = 5)")
        outcome = meta_mod.dispatch_meta(":bindings", _session_ctx(s))
        assert outcome.text is not None
        assert "Point(\n  y = 5,\n  x = 3\n)" in outcome.text

    def test_load_persists_record_binding_for_bindings_rendering(self, tmp_path: Path) -> None:
        src = tmp_path / "rec.agl"
        src.write_text("record Point\n  y: int\n  x: int\nlet p = Point(x = 1, y = 2)\n")
        s = _open_session()
        outcome = meta_mod.dispatch_meta(f":load {src}", _session_ctx(s))

        assert outcome.text == "Point declared\np : Point = Point(\n  y = 2,\n  x = 1\n)"
        bindings = meta_mod.dispatch_meta(":bindings", _session_ctx(s))
        assert bindings.text is not None
        assert "Point(\n  y = 2,\n  x = 1\n)" in bindings.text


class TestScopedDeclarationEcho:
    """Verify scoped declarations echo the scope they landed in."""

    def test_scope_region_entry_echoes_its_path(self) -> None:
        s = _open_session()
        r = s.eval_entry("scope Tools\n  def twice(x: int) -> int = x * 2\nend Tools")

        assert r.ok, r.diagnostics
        assert render_mod.render_entry_result(r, echo=True) == "Tools declared"

    def test_check_only_scope_region_entry_echoes_its_path(self) -> None:
        s = _open_session()
        r = s.eval_entry(
            "scope Tools\n  def twice(x: int) -> int = x * 2\nend Tools", check_only=True
        )

        assert r.ok, r.diagnostics
        assert render_mod.render_entry_result(r, echo=True, check_only=True) == "Tools declared"

    def test_empty_scope_region_entry_echoes_its_path(self) -> None:
        s = _open_session()
        r = s.eval_entry("scope Tools\nend Tools")

        assert r.ok, r.diagnostics
        assert render_mod.render_entry_result(r, echo=True) == "Tools declared"

    def test_multi_segment_scope_region_entry_echoes_the_full_path(self) -> None:
        s = _open_session()
        r = s.eval_entry("scope Outer::Inner\n  def value() -> int = 1\nend Outer::Inner")

        assert r.ok, r.diagnostics
        assert render_mod.render_entry_result(r, echo=True) == "Outer::Inner declared"

    def test_region_with_its_own_member_echoes_only_the_outer_path(self) -> None:
        s = _open_session()
        r = s.eval_entry(
            "scope Outer\n  def value() -> int = 1\n\n  scope Inner\n  end Inner\nend Outer",
        )

        assert r.ok, r.diagnostics
        assert render_mod.render_entry_result(r, echo=True) == "Outer declared"

    def test_scoped_shorthand_declaration_echoes_its_full_path(self) -> None:
        s = _open_session()
        func = s.eval_entry("def Tools::twice(x: int) -> int = x * 2")
        rec = s.eval_entry("record Tools::Pair(left: int, right: int)")

        assert func.ok, func.diagnostics
        assert rec.ok, rec.diagnostics
        assert render_mod.render_entry_result(func, echo=True) == "Tools::twice declared"
        assert render_mod.render_entry_result(rec, echo=True) == "Tools::Pair declared"

    def test_root_declaration_echo_stays_unqualified(self) -> None:
        s = _open_session()
        r = s.eval_entry("record Pair(left: int, right: int)")

        assert r.ok, r.diagnostics
        assert render_mod.render_entry_result(r, echo=True) == "Pair declared"

    def test_exception_declaration_echoes_its_name(self) -> None:
        s = _open_session()
        r = s.eval_entry("exception Boom extends Exception\n  code: int")

        assert r.ok, r.diagnostics
        assert render_mod.render_entry_result(r, echo=True) == "Boom declared"

    def test_use_declaration_entry_echoes_nothing(self) -> None:
        s = _open_session()
        assert s.eval_entry("def Tools::twice(x: int) -> int = x * 2").ok
        r = s.eval_entry("use Tools::*")

        assert r.ok, r.diagnostics
        assert render_mod.render_entry_result(r, echo=True) is None
