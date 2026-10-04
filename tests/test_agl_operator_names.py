"""Operators are names, and their fixity belongs to the module defining them.

An infix operator spelling is looked up exactly like a function name: a
lexical binding is nearest, the module's own declaration beats an imported
one, two distinct imported declarations are ambiguous, and scope regions,
``use``, import tails, re-exports and renames reach it as they reach any
function. The fixity applied is the one the selected declaration's module
declares for that declaration's name -- at any scope path, a lexical binder
included -- and selecting a declaration whose module declares none is an
error at the operator. A fixity declaration for a name its module declares
nowhere is an error at the declaration; in the REPL the session is the
module, so the declaration comes with or after its definition.

``at prio Y`` reads ``Y`` as an operator spelling at the module root, the
module's own fixity declarations first in any order: a builtin operator, an
own operator or an imported one gives its fixity as the base.

Every probe is checked in the file part and in every legal REPL grouping of its
scenario's header (see :mod:`tests.agl.qualifier_support`).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agm.agl.diagnostics import AglTypeError
from agm.agl.parser import AglSyntaxError
from agm.agl.scope.symbols import AglScopeError, AmbiguousQualificationError
from tests._timeouts import fail_if_slow
from tests.agl.qualifier_support import (
    Probe,
    Scenario,
    accepted,
    all_groupings,
    assert_repl_verdicts,
    assert_scenario,
    inline_verdict,
    rejected,
    scenario_params,
)


def _t(name: str, left: str, right: str) -> str:
    """The spelling of the two-field record type ``name[left, right]``."""
    return f"{name}[{left}, {right}]"


def _pair(name: str, left: str, right: str) -> str:
    """The rendered identity of a value of the two-field record type ``name[left, right]``."""
    return f"record {_t(name, left, right)}\n  l: {left}\n  r: {right}"


def _builder(name: str, record: str) -> str:
    """A generic operator *name* pairing its operands into *record*."""
    return f"def {name}[A, B](a: A, b: B) -> {record}[A, B] = {record}(a, b)"


def _record(name: str) -> str:
    return f"record {name}[A, B]\n  l: A\n  r: B"


_P = _record("P")
_Q = _record("Q")
_OPS = f"{_record('L')}\n\ninfixl <+> at 5\n\n{_builder('<+>', 'L')}\n"
_OTHER_OPS = f"{_record('M')}\n\ninfixr <+> at 7\n\n{_builder('<+>', 'M')}\n"
_NO_FIXITY = "def <+>(a: int, b: int) -> int = a\n"
_INT = "int"
_L_INT = _t("ops::L", _INT, _INT)


_ROOT_LAMBDA = "fn(a: int, b: int) -> int => a * 10 + b"


def _root_binder(probe: Probe) -> Probe:
    """*probe* after a root binder of :data:`_ROOT_LAMBDA`, which a file rejects as no constant."""
    return replace(
        probe, in_file=rejected(probe.text, AglTypeError, _ROOT_LAMBDA, phase="typecheck")
    )


def _same_entry(sizes: tuple[int, ...], first: int, second: int) -> bool:
    """Whether items *first* and *second* fall in one entry of grouping *sizes*."""
    end = 0
    for size in sizes:
        end += size
        if first < end:
            return second < end
    return False


def _together(n: int, first: int, second: int) -> frozenset[tuple[int, ...]]:
    """Every grouping of *n* items keeping items *first* and *second* in one entry."""
    return frozenset(sizes for sizes in all_groupings(n) if _same_entry(sizes, first, second))


_SCENARIOS: dict[str, Scenario] = {
    "visibility": Scenario(
        header=("import ops::*",),
        modules={"ops": _OPS},
        probes={
            "tail": accepted("1 <+> 2 <+> 3", _pair("ops::L", _L_INT, _INT)),
            "builtin-operands": accepted("1 + 2 <+> 3 * 4", _pair("ops::L", _INT, _INT)),
            "lexical-shadow-without-fixity": rejected(
                "let <+> = fn(a: int, b: int) -> int => a\n1 <+> 2", AglSyntaxError, "<+>"
            ),
            "fixity-for-imported": rejected(
                "infixr <+> at 9\n1 <+> 2", AglSyntaxError, "infixr <+> at 9"
            ),
            "scoped-wildcard-import": accepted(
                "scope Local\n  import ops::*\n"
                "  def apply(x: int, y: int) -> L[int, int] = x <+> y\nend Local\n"
                "\n"
                "Local::apply(1, 2)",
                _pair("ops::L", _INT, _INT),
            ),
        },
    ),
    "qualified-import": Scenario(
        header=("import ops",),
        modules={"ops": _OPS},
        probes={
            "not-bare": rejected("1 <+> 2", AglScopeError, "<+>"),
            "use-rename": accepted(
                "use ops::{<+> as <&>}\n1 <&> 2 <&> 3", _pair("ops::L", _L_INT, _INT)
            ),
        },
    ),
    "region-scoped-import": Scenario(
        header=(
            "scope Local\n  import ops::*\n"
            "  def apply(x: int, y: int) -> L[int, int] = x <+> y\nend Local",
        ),
        modules={"ops": _OPS},
        probes={"outside-region": rejected("1 <+> 2", AglScopeError, "<+>")},
    ),
    "own-beats-import": Scenario(
        header=("import ops::*", _P, _builder("<+>", "P"), "infixr <+> at 5"),
        modules={"ops": _OPS},
        probes={"own": accepted("1 <+> 2 <+> 3", _pair("P", _INT, _t("P", _INT, _INT)))},
    ),
    "two-imports": Scenario(
        header=("import ops::*", "import other::*"),
        modules={"ops": _OPS, "other": _OTHER_OPS},
        probes={
            "ambiguous": rejected("1 <+> 2", AmbiguousQualificationError, "<+>"),
            "ambiguous-priority-base": rejected(
                f"infixl <*> at prio <+> + 1\n{_builder('<*>', 'L')}",
                AmbiguousQualificationError,
                "infixl <*> at prio <+> + 1",
            ),
        },
    ),
    "reexports": Scenario(
        header=(),
        modules={
            "ops": _OPS,
            "renamed": "export ops::{<+> as <&>}\n",
            "public": "scope Public\n  export ops::{<+>}\nend Public\n",
            "forwarding": "scope Public\n  export ops\nend Public\n",
            "hidden": "export ops hiding <+>\n",
        },
        probes={
            "renamed-keeps-defining-fixity": accepted(
                "import renamed::*\n1 <&> 2 <&> 3", _pair("ops::L", _L_INT, _INT)
            ),
            "used-scoped-reexport": accepted(
                "import public::*\nuse Public::*\n1 <+> 2", _pair("ops::L", _INT, _INT)
            ),
            "scoped-reexport-not-bare": rejected("import public::*\n1 <+> 2", AglScopeError, "<+>"),
            "untailed-scoped-reexport": accepted(
                "import forwarding::*\nuse Public::*\n1 <+> 2", _pair("ops::L", _INT, _INT)
            ),
            "aliased-module-use-rename": accepted(
                "import public as API\nuse API::Public::{<+> as <&>}\n1 <&> 2 <&> 3",
                _pair("ops::L", _L_INT, _INT),
            ),
            "hidden-reexport": rejected("import hidden::*\n1 <+> 2", AglScopeError, "<+>"),
            "region-import-rename": accepted(
                "scope Local\n  import ops\n  import public::{Public::<+> as <+>}\n"
                "  def apply(x: int, y: int) -> ops::L[int, int] = x <+> y\nend Local\n"
                "\n"
                "Local::apply(1, 2)",
                _pair("ops::L", _INT, _INT),
            ),
        },
    ),
    "missing-fixity": Scenario(
        header=("import nofix::*",),
        modules={"nofix": _NO_FIXITY},
        probes={
            "operator": rejected("1 <+> 2", AglSyntaxError, "<+>"),
            "priority-base": rejected(
                "infixl <*> at prio <+> + 1\ndef <*>(a: int, b: int) -> int = a\n1 <*> 2",
                AglSyntaxError,
                "infixl <*> at prio <+> + 1",
            ),
        },
    ),
    "scoped-definition": Scenario(
        header=(_P, f"scope S\n  {_builder('<+>', 'P')}\nend S", "infixr <+> at 5"),
        probes={
            "root-not-visible": rejected("1 <+> 2", AglScopeError, "<+>"),
            "used": accepted(
                "scope U\n  use S::*\n  def h() -> P[int, P[int, int]] = 1 <+> 2 <+> 3\nend U\n"
                "\n"
                "U::h()",
                _pair("P", _INT, _t("P", _INT, _INT)),
            ),
            "member-body": accepted(
                "def S::g() -> P[int, P[int, int]] = 1 <+> 2 <+> 3\nS::g()",
                _pair("P", _INT, _t("P", _INT, _INT)),
            ),
        },
    ),
    "standalone": Scenario(
        header=(_P, _Q),
        probes={
            "lexical-binders-in-body": accepted(
                "infixl <+> at 5\ninfixl <*> at 6\n"
                "def f() -> text =\n"
                '  let <+> = fn(a: int, b: bool) -> text => "t"\n'
                "  let <*> = fn(a: int, b: int) -> bool => true\n"
                "  1 <+> 2 <*> 3\n"
                "f()",
                "text",
            ),
            "declared-nowhere": rejected("infixl <+> at 5\n()", AglSyntaxError, "infixl <+> at 5"),
            "redeclared-fixity": rejected(
                f"infixl <+> at 5\ninfixr <+> at 6\n{_builder('<+>', 'P')}",
                AglSyntaxError,
                "infixr <+> at 6",
            ),
            "builtin-redeclared": rejected("infixl + at 5\n()", AglSyntaxError, "infixl + at 5"),
            "mixed-associativity": rejected(
                f"infixl <+> at 5\ninfixr <*> at 5\n{_builder('<+>', 'P')}\n"
                f"{_builder('<*>', 'Q')}\n1 <+> 2 <*> 3",
                AglSyntaxError,
                "<*>",
            ),
            "priority-later-own": accepted(
                f"infixl <*> at prio <+> + 1\ninfixl <+> at 5\n{_builder('<+>', 'P')}\n"
                f"{_builder('<*>', 'Q')}\n1 <+> 2 <*> 3",
                _pair("P", _INT, _t("Q", _INT, _INT)),
            ),
            "default-priority-is-additive": rejected(
                f"infixl <+>\ninfixr <*> at prio + + 0\n{_builder('<+>', 'P')}\n"
                f"{_builder('<*>', 'Q')}\n1 <+> 2 <*> 3",
                AglSyntaxError,
                "<*>",
            ),
            "priority-builtin": accepted(
                f"infixl <+> at prio + - 1\n{_builder('<+>', 'P')}\n1 + 2 <+> 3",
                _pair("P", _INT, _INT),
            ),
            "priority-cycle": rejected(
                "infixl <+> at prio <*> + 1\ninfixl <*> at prio <+> + 1\n"
                f"{_builder('<+>', 'P')}\n{_builder('<*>', 'Q')}",
                AglSyntaxError,
                "infixl <+> at prio <*> + 1",
            ),
            "priority-unknown": rejected(
                f"infixl <+> at prio ?? + 1\n{_builder('<+>', 'P')}",
                AglSyntaxError,
                "infixl <+> at prio ?? + 1",
            ),
            "not-prefix": accepted(
                "infixl <+> at 5\ndef <+>(a: bool, b: bool) -> bool = a\nnot true <+> false", "bool"
            ),
            "attribute-argument": rejected(
                f"infixl <+> at 5\n{_builder('<+>', 'P')}\n@doc(1 <+> 2)\ndef g() -> int = 1",
                AglScopeError,
                "1 <+> 2",
            ),
            "parenthesized-comparison": accepted("(1 == 1) == true", "bool"),
            "parenthesized-comparison-right": accepted("true == (1 == 1)", "bool"),
        },
    ),
    "imported-priority-base": Scenario(
        header=("import ops::*", _P, _Q),
        modules={"ops": _OPS},
        probes={
            "above": accepted(
                f"infixl <*> at prio <+> + 1\n{_builder('<*>', 'P')}\n1 <+> 2 <*> 3",
                _pair("ops::L", _INT, _t("P", _INT, _INT)),
            ),
            "below": accepted(
                f"infixl <*> at prio <+> - 1\n{_builder('<*>', 'P')}\n1 <+> 2 <*> 3",
                _pair("P", _L_INT, _INT),
            ),
            "own-base-first": accepted(
                f"infixl <*> at prio <+> + 1\ninfixl <+> at 3\n{_builder('<+>', 'P')}\n"
                f"{_builder('<*>', 'Q')}\n1 <*> 2 <+> 3",
                _pair("P", _t("Q", _INT, _INT), _INT),
            ),
        },
    ),
    "session-root-let": Scenario(
        header=(f"let <+> = {_ROOT_LAMBDA}",),
        probes={"fixity-and-chain": _root_binder(accepted("infixl <+> at 5\n1 <+> 2 <+> 3", _INT))},
    ),
    "session-root-var": Scenario(
        header=(f"var <+> = {_ROOT_LAMBDA}",),
        probes={"fixity-and-chain": _root_binder(accepted("infixr <+> at 5\n1 <+> 2 <+> 3", _INT))},
    ),
    "session-root-let-then-fixity": Scenario(
        header=(f"let <+> = {_ROOT_LAMBDA}", "infixl <+> at 5"),
        probes={"chain": _root_binder(accepted("1 <+> 2 <+> 3", _INT))},
    ),
    "definition-first": Scenario(
        header=(_P, _builder("<+>", "P"), "infixr <+> at 5"),
        probes={"chain": accepted("1 <+> 2 <+> 3", _pair("P", _INT, _t("P", _INT, _INT)))},
    ),
    "fixity-first": Scenario(
        header=(_P, "infixr <+> at 5", _builder("<+>", "P")),
        legal=_together(4, 1, 2),
        probes={"chain": accepted("1 <+> 2 <+> 3", _pair("P", _INT, _t("P", _INT, _INT)))},
    ),
}


class TestOperatorNames:
    @pytest.mark.parametrize("scenario", scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)


class TestSessionFixity:
    """The session is the module: fixity declared once holds for later entries."""

    def test_later_fixity_declaration_replaces_the_earlier_one(self, tmp_path: Path) -> None:
        assert_repl_verdicts(
            tmp_path,
            {},
            (_P, _builder("<+>", "P"), "infixl <+> at 5", "infixr <+> at 5"),
            {"chain": accepted("1 <+> 2 <+> 3", _pair("P", _INT, _t("P", _INT, _INT)))},
        )

    def test_redefined_operator_keeps_the_session_fixity(self, tmp_path: Path) -> None:
        assert_repl_verdicts(
            tmp_path,
            {},
            (_P, _Q, f"infixr <+> at 5\n{_builder('<+>', 'P')}", _builder("<+>", "Q")),
            {"chain": accepted("1 <+> 2 <+> 3", _pair("Q", _INT, _t("Q", _INT, _INT)))},
        )

    def test_body_local_binder_declares_nothing_for_a_later_entry(self, tmp_path: Path) -> None:
        assert_repl_verdicts(
            tmp_path,
            {},
            (f"def f() -> int =\n  let <+> = {_ROOT_LAMBDA}\n  1",),
            {"fixity": rejected("infixl <+> at 5", AglSyntaxError, "infixl <+> at 5")},
        )

    def test_fixity_for_an_imported_operator_is_rejected(self, tmp_path: Path) -> None:
        assert_repl_verdicts(
            tmp_path,
            {"ops": _OPS},
            ("import ops::*",),
            {"fixity": rejected("infixr <+> at 9", AglSyntaxError, "infixr <+> at 9")},
        )


class TestImportedModuleFixity:
    """An imported module's fixity declarations are checked like the entry's own."""

    def _rejection(self, tmp_path: Path, modules: dict[str, str]) -> tuple[object, str]:
        phase, error, span, _identity = inline_verdict(tmp_path, modules)
        assert span is not None
        return (phase, error), Path(span.source.label).stem

    def test_fixity_declared_nowhere_is_rejected(self, tmp_path: Path) -> None:
        modules = {"fixonly": "infixl <+> at 5\n", "entry": "import fixonly\n()"}

        assert self._rejection(tmp_path, modules) == (("scope", AglSyntaxError), "fixonly")

    def test_unknown_priority_base_is_rejected(self, tmp_path: Path) -> None:
        modules = {
            "badprio": f"{_record('L')}\ninfixl <+> at prio ?? + 1\n{_builder('<+>', 'L')}\n",
            "entry": "import badprio\n()",
        }

        assert self._rejection(tmp_path, modules) == (("scope", AglSyntaxError), "badprio")

    def test_cyclic_scoped_reexports_are_rejected_without_hanging(self, tmp_path: Path) -> None:
        modules = {
            "ops": _OPS,
            "a": "export ops\n\nscope Loop\n  export b\nend Loop\n",
            "b": "scope Loop\n  export a\nend Loop\n",
            "entry": "import a::*\n1 <+> 2",
        }

        with fail_if_slow("operator re-export resolution did not terminate"):
            phase, error, _span, _identity = inline_verdict(tmp_path, modules)

        assert (phase, error) == ("scope", AglScopeError)
