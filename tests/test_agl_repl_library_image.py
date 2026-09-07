"""REPL workflows stay correct across imports, resets, and redeclarations."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.repl import ReplSession
from agm.agl.semantics.values import BoolValue, IntValue, TextValue


def _open_session(root: Path | None = None) -> ReplSession:
    session = ReplSession(cwd=root, extra_cli_roots=() if root is None else (str(root),))
    assert session.open() == ()
    return session


@pytest.mark.parametrize(
    ("entry", "expected"),
    [("x + 1", 2), ("case x of\n  | 1 => 2\n  | _ => 3", 2), ("x + 2 * 3", 7)],
    ids=["binding", "match", "precedence"],
)
def test_later_entries_use_prior_bindings(entry: str, expected: int) -> None:
    session = _open_session()
    assert session.eval_entry("let x = 1").ok

    result = session.eval_entry(entry)

    assert result.ok, result.diagnostics
    assert result.value == IntValue(expected)


def test_reset_discards_bindings_and_accepts_fresh_definitions() -> None:
    session = _open_session()
    assert session.eval_entry("let x = 1").ok
    session.reset()

    missing = session.eval_entry("x")
    assert not missing.ok
    assert missing.diagnostics
    assert session.eval_entry("let x = 7").ok
    assert session.eval_entry("x + 1").value == IntValue(8)


class TestImportedModules:
    def test_a_newly_imported_module_is_resolved_and_used(self, tmp_path: Path) -> None:
        (tmp_path / "later.agl").write_text("def twice(n: int) -> int = n * 2\n")
        session = _open_session(tmp_path)
        assert session.eval_entry("let x = 1").ok

        result = session.eval_entry("import later::*\ntwice(21)")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(42)

    def test_imported_operators_evaluate_with_declared_precedence(self, tmp_path: Path) -> None:
        (tmp_path / "ops.agl").write_text(
            "infixl <+> at 6\n"
            "def <+>(a: int, b: int) -> int = a + b + 1\n"
            "def combined() -> int = 1 <+> 2 <+> 3\n"
        )
        session = _open_session(tmp_path)
        assert session.eval_entry("let x = 1").ok

        result = session.eval_entry("import ops::*\ncombined()")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(8)

    def test_a_later_import_of_a_second_module_still_sees_the_first(self, tmp_path: Path) -> None:
        (tmp_path / "one.agl").write_text("def one() -> int = 1\n")
        (tmp_path / "two.agl").write_text("def two() -> int = 2\n")
        session = _open_session(tmp_path)
        assert session.eval_entry("import one::*\none()").ok

        result = session.eval_entry("import two::*\none() + two()")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(3)

    def test_sessions_with_different_roots_use_their_own_module(self, tmp_path: Path) -> None:
        first = tmp_path / "first"
        first.mkdir()
        second = tmp_path / "second"
        second.mkdir()
        body = "def pick(n: int) -> int =\n  case n of\n    | 0 => %s\n    | _ => %s\n"
        (first / "shared.agl").write_text(body % (1, 2))
        (second / "shared.agl").write_text(body % (30, 40))

        session = _open_session(first)
        assert session.eval_entry("import shared::*\npick(0)").value == IntValue(1)

        session = _open_session(second)
        result = session.eval_entry("import shared::*\npick(0)")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(30)

    def test_a_reset_session_rereads_a_module_whose_file_changed(self, tmp_path: Path) -> None:
        """Reuse survives a reset only for modules the reopened session reloads."""
        module = tmp_path / "shifting.agl"
        module.write_text("def answer() -> int = 1\n")
        session = _open_session(tmp_path)
        assert session.eval_entry("import shifting::*\nanswer()").value == IntValue(1)

        session.reset()
        module.write_text("def answer() -> int = 2\n")
        result = session.eval_entry("import shifting::*\nanswer()")

        assert result.ok, result.diagnostics
        assert result.value == IntValue(2)


class TestRedeclarations:
    def test_redeclared_enum_keeps_old_and_new_member_identities_apart(self) -> None:
        session = _open_session()
        assert session.eval_entry("enum E\n  | A(old: int)").ok
        assert session.eval_entry("type OldA = E::A").ok
        assert session.eval_entry("let old: E = A(old = 1)").ok
        assert session.eval_entry("enum E\n  | B(fresh: text)").ok
        assert session.eval_entry('let new: E = B(fresh = "new")').ok

        assert session.eval_entry("old is A").value == BoolValue(True)
        assert session.eval_entry("(old as OldA).old").value == IntValue(1)
        assert session.eval_entry("(new as E::B).fresh").value == TextValue("new")
        assert not session.eval_entry("old as E::B").ok

    def test_redeclared_record_keeps_the_retained_value_at_its_old_shape(self) -> None:
        session = _open_session()
        assert session.eval_entry("record R\n  a: int").ok
        assert session.eval_entry("let old = R(a = 1)").ok
        assert session.eval_entry("record R\n  b: text").ok

        assert session.eval_entry("old.a").value == IntValue(1)
        assert not session.eval_entry("let mixed: R = old").ok
        assert session.eval_entry('R(b = "x").b').value == TextValue("x")


def test_session_operators_do_not_change_imported_function_behavior(tmp_path: Path) -> None:
    (tmp_path / "ops.agl").write_text(
        "infixl <+> at 6\n"
        "def <+>(a: int, b: int) -> int = a + b + 1\n"
        "infixr <?> at 7\n"
        "def <?>(a: int, b: int) -> int = a * b + 2\n"
        "def combined() -> int = 1 <+> 2 <?> 3\n"
    )
    session = _open_session(tmp_path)
    assert session.eval_entry("import ops::*\ncombined()").value == IntValue(10)
    assert session.eval_entry("1 <+> 2 <?> 3 <+> 4").value == IntValue(15)

    assert session.eval_entry("infixr <+> at 9\ndef <+>(a: int, b: int) -> int = a - b").ok

    assert session.eval_entry("combined()").value == IntValue(10)
    ambiguous = session.eval_entry("8 <+> 3 <+> 1")
    assert not ambiguous.ok
    assert ambiguous.diagnostics
    assert session.eval_entry("combined()").value == IntValue(10)


@pytest.mark.parametrize(
    ("bad_entry", "expected_error"),
    [('let x: int = "bad"', "static"), ("let ( = 1", "static"), ("1 / 0", "runtime")],
    ids=["type-error", "syntax-error", "runtime-error"],
)
def test_failed_entries_preserve_prior_state_and_allow_recovery(
    bad_entry: str, expected_error: str
) -> None:
    session = _open_session()
    assert session.eval_entry("let x = 21").ok

    failed = session.eval_entry(bad_entry)

    assert not failed.ok
    if expected_error == "static":
        assert failed.diagnostics
    else:
        assert failed.error is not None
    assert session.eval_entry("x * 2").value == IntValue(42)
