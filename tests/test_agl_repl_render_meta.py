"""Unit tests for the REPL renderer and meta-command dispatcher.

These are pure-data modules (no terminal), so they are tested directly:

- :func:`agm.agl.repl.render.render_entry_result` for each entry kind, warnings,
  pre-execution diagnostics, and runtime-error rendering;
- :func:`agm.agl.repl.meta.dispatch_meta` for ``:help`` / ``:quit`` / ``:exit`` /
  unknown commands, plus the ``register_meta_command`` extension hook used by M3.
"""

from __future__ import annotations

from decimal import Decimal

from agm.agl.diagnostics import Diagnostic
from agm.agl.eval.values import IntValue, TextValue, Value
from agm.agl.repl import meta as meta_mod
from agm.agl.repl import render as render_mod
from agm.agl.repl.session import EntryKind, EntryResult, ReplSession
from agm.agl.runtime.runtime import RunError
from agm.agl.typecheck.types import IntType, TextType, Type


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
) -> EntryResult:
    """Build an ``EntryResult`` with sensible defaults for one test axis."""
    return EntryResult(
        kind=kind,
        name=name,
        value=value,
        value_type=value_type,
        diagnostics=diagnostics if diagnostics is not None else [],
        warnings=warnings if warnings is not None else [],
        error=error,
        ok=ok,
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

    def test_declaration_echo(self) -> None:
        result = _result(kind="declaration", name="R", ok=True)
        assert render_mod.render_entry_result(result, echo=True) == "R declared"

    def test_statement_echo_is_none(self) -> None:
        result = _result(kind="statement", ok=True)
        assert render_mod.render_entry_result(result, echo=True) is None

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
        result = _result(
            kind="expression", value=TextValue("hi"), value_type=TextType(), ok=True
        )
        assert render_mod.render_entry_result(result, echo=False) is None

    def test_warnings_are_always_rendered(self) -> None:
        result = _result(
            kind="statement",
            ok=True,
            warnings=[Diagnostic(message="watch out", line=2)],
        )
        rendered = render_mod.render_entry_result(result, echo=False)
        assert rendered == "warning: line 2: watch out"

    def test_pre_execution_diagnostics(self) -> None:
        result = _result(
            ok=False,
            diagnostics=[Diagnostic(message="boom", line=1)],
        )
        assert render_mod.render_entry_result(result, echo=True) == "line 1: boom"

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
            warnings=[Diagnostic(message="w", line=1)],
            diagnostics=[Diagnostic(message="e", line=2)],
        )
        rendered = render_mod.render_entry_result(result, echo=True)
        assert rendered == "warning: line 1: w\nline 2: e"


# ---------------------------------------------------------------------------
# dispatch_meta
# ---------------------------------------------------------------------------


def _ctx() -> meta_mod.MetaContext:
    return meta_mod.MetaContext(session=ReplSession())


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

    def test_register_meta_command_extends_registry(self) -> None:
        # M3 extension hook: a newly registered command becomes dispatchable and
        # is offered by ``meta_command_names`` (the completer's source).
        seen: list[str] = []

        def handler(arg: str, ctx: meta_mod.MetaContext) -> meta_mod.MetaOutcome:
            seen.append(arg)
            return meta_mod.MetaOutcome(text="did thing")

        command = meta_mod.MetaCommand(
            names=("xtest",),
            usage=":xtest",
            summary="A test command.",
            handler=handler,
        )
        meta_mod.register_meta_command(command)
        try:
            outcome = meta_mod.dispatch_meta(":xtest arg1", _ctx())
            assert outcome.text == "did thing"
            assert seen == ["arg1"]
            assert ":xtest" in meta_mod.meta_command_names()
        finally:
            meta_mod._COMMANDS.remove(command)
