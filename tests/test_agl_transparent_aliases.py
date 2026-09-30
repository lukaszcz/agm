"""An alias is another name for its target declaration.

A type alias, ``import m as lib``, ``use … as`` and a one-segment ``use L as
M`` add spellings, never declarations: a spelling through an alias means what
the target's spelling means at the alias's site, in every position and for
every path beneath the target. Reaching one declaration through an alias and
directly, or through two aliases, is one declaration; two distinct targets
still clash. An importer's ``hiding`` removes the declaration it names -- and
the paths beneath it -- from every spelling that import brings in, aliases
the imported module declares included. A tail or ``hiding`` item may name
a path through an alias: it names the target's declaration there, and a
path naming none is an error. An alias whose target resolves to nothing is
an error where it is declared.

Every probe is checked in file mode and in every legal REPL grouping of its
scenario's header (see :mod:`tests.agl.qualifier_support`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import AglTypeError, HiddenMemberError
from agm.agl.scope.symbols import (
    AglScopeError,
    AmbiguousConstructorError,
    AmbiguousQualificationError,
    NoVisibleConstructorError,
    TypeArgumentsError,
    UnknownMemberError,
)
from tests.agl.qualifier_support import (
    Probe,
    Scenario,
    accepted,
    assert_file_resolves_like_inline_entry,
    assert_scenario,
    file_params,
    file_verdict,
    info,
    rejected,
    scenario_params,
    span_text,
    type_positions,
    type_positions_rejected,
)

_BASE = (
    "record Base\n  x: int\nrecord Base::Inner\n  y: int\n"
    "def Base::f() -> int = 1\nenum Base::Shape\n  | Sq\n  | Ci\n"
    "exception Oops\n  code: int\nrecord Box[T]\n  v: T\nrecord Box::In\n  w: int\n"
    "enum Color\n  | Red\n  | Green\n"
)
_EXPORTING = (
    "import base\nexport base::{Base, Color}\ntype Geo = base::Base\n"
    "type Sh = base::Base::Shape\ntype Uh = base::Oops\ntype Bx[T] = base::Box[T]\n"
    "type Hue = base::Color\n"
)
"""Exports ``base``'s ``Base`` and ``Color``, and aliases of ``base``'s declarations."""
_RENAMING = "import base\ntype Base = base::Base\ntype Geo = Base\n"
"""Declares an alias named like its target, and an alias of that alias."""
_TWIN = "import base\ntype Geo = base::Base\n"
_ELSEWHERE = "record Base\n  x: int\ntype Geo = Base\n"
_REEXPORTING = "import al\nexport al::{Geo}\n"
_EXPORTING_THROUGH = (
    "import al\nexport al::{Geo::Inner, Geo::Inner as I, Geo::Shape, Geo::f, Hue::Red}\n"
)
"""Re-exports paths written through ``al``'s aliases."""
_HIDING_THROUGH = "import al\nexport al hiding Geo::Inner\n"
_MODULES = {"base": _BASE, "al": _EXPORTING}

_RECORD = "record base::Base\n  x: int"
_INNER = "record base::Base::Inner\n  y: int"
_GREEN = "record base::Color::Green"
_BASE_CASE = "fn(p: base::Base) => case p of\n    | Geo(x) => x"
"""Matches ``base``'s ``Base`` by ``Geo`` in a region (see :func:`_in_region`)."""


def _in_region(use: str, value: str) -> str:
    """A region whose ``use`` exposes names that its ``let v`` reads, then ``r::v``."""
    return f"scope r\n  {use}\n  let v = {value}\nend r\n\nr::v"


def _one_declaration_probes(spelling: str, scrutinee: str = "base::Base") -> dict[str, Probe]:
    """*spelling* selecting ``base``'s ``Base`` although several sources reach it.

    *scrutinee* spells ``Base`` without them.
    """
    return {
        **type_positions(spelling, spelling, "base::Base"),
        f"{spelling}-value": accepted(f"{spelling}(x = 1)", _RECORD),
        f"{spelling}-pattern": accepted(
            f"case {scrutinee}(x = 1) of\n  | {spelling}(x) => x", "int"
        ),
        f"{spelling}-nested": accepted(f"{spelling}::Inner(y = 1)", _INNER),
        f"{spelling}-function": accepted(f"{spelling}::f", "() -> int"),
    }


def _hidden_probes(spelling: str) -> dict[str, Probe]:
    """*spelling*, a path an import's ``hiding`` removed, rejected in every position."""
    return {
        f"{spelling}-value": rejected(f"{spelling}(y = 1)", HiddenMemberError, spelling),
        **type_positions_rejected(spelling, spelling, HiddenMemberError),
    }


_SCENARIOS = {
    "an-alias-named-like-its-target-is-that-declaration": Scenario(
        modules={"base": _BASE, "al": _RENAMING},
        header=("import base::*", "import al::*"),
        probes={
            **_one_declaration_probes("Base"),
            "info": info(
                "Base", "Base is a constructor.\nSignature:\n  Base(x: int) -> base::Base"
            ),
        },
    ),
    "an-alias-chain-and-its-target-are-one-declaration": Scenario(
        modules={"base": _BASE, "al": _RENAMING, "twin": "import base::*\ntype Geo = Base\n"},
        header=("import al::*", "import twin::*"),
        probes=_one_declaration_probes("Geo", "al::Geo"),
    ),
    "two-aliases-of-one-target-are-one-declaration": Scenario(
        modules={"base": _BASE, "al": _EXPORTING, "twin": _TWIN},
        header=("import al::*", "import twin::*"),
        probes=_one_declaration_probes("Geo", "al::Geo"),
    ),
    "a-re-exported-alias-and-its-target-are-one-declaration": Scenario(
        modules={**_MODULES, "re": _REEXPORTING},
        header=("import re::*", "import base::*"),
        probes={
            **_one_declaration_probes("Geo"),
            **_one_declaration_probes("Base"),
        },
    ),
    "aliases-of-distinct-targets-clash": Scenario(
        modules={**_MODULES, "other": _ELSEWHERE},
        header=("import al::*", "import other::*"),
        probes={
            "value": rejected("Geo(x = 1)", AmbiguousConstructorError, "Geo"),
            **type_positions_rejected("type", "Geo", AmbiguousQualificationError),
            "nested-only-one-declares": accepted("Geo::Inner(y = 1)", _INNER),
        },
    ),
    "hiding-a-target-path-hides-it-through-the-imported-module-aliases": Scenario(
        modules=_MODULES,
        header=("import al::* hiding Base::Inner",),
        probes={
            **_hidden_probes("Geo::Inner"),
            **_hidden_probes("Base::Inner"),
            "routed": rejected("al::Geo::Inner(y = 1)", HiddenMemberError, "al::Geo::Inner"),
            "alias": accepted("Geo(x = 1)", _RECORD),
            "sibling": accepted("Geo::f", "() -> int"),
        },
    ),
    "hiding-leaves-what-other-imports-bring": Scenario(
        modules=_MODULES,
        # One entry: a later REPL entry's import of ``al`` supersedes an earlier one.
        header=("import al::* hiding Base::Inner\nimport al", "import base::*"),
        probes={
            "other-import": accepted("Base::Inner(y = 1)", _INNER),
            "routed-by-another-import": accepted("al::Geo::Inner(y = 1)", _INNER),
            **_hidden_probes("Geo::Inner"),
        },
    ),
    "hiding-a-target-removes-its-aliases": Scenario(
        modules=_MODULES,
        header=("import al::* hiding Base", "import base"),
        probes={
            "alias-value": rejected("Geo(x = 1)", AglScopeError, "Geo"),
            "alias-pattern": rejected(
                "fn(p: base::Base) => case p of\n  | Geo(x) => x",
                NoVisibleConstructorError,
                "Geo(x)",
            ),
            "alias-annot": rejected("fn(p: Geo) => p", AglTypeError, "p: Geo", phase="typecheck"),
            "alias-alias": rejected("type AA = Geo\nfn(p: AA) => p", AglTypeError, "type AA = Geo"),
            "alias-tyarg": rejected(
                "fn(p: array[Geo]) => p", AglTypeError, "Geo", phase="typecheck"
            ),
            "alias-cast": rejected(
                "fn(p: text) => p as? Geo", AglTypeError, "p as? Geo", phase="typecheck"
            ),
            "beneath-the-alias": rejected("Geo::Inner(y = 1)", HiddenMemberError, "Geo::Inner"),
            "routed-alias": rejected("al::Geo(x = 1)", HiddenMemberError, "al::Geo"),
            "beneath-it-through-another-alias": rejected(
                "fn(p: Sh) => p", AglTypeError, "p: Sh", phase="typecheck"
            ),
            "unrelated-alias": accepted("fn(p: Uh) => p", "base::Oops -> base::Oops"),
        },
    ),
    "hiding-names-a-path-through-an-alias": Scenario(
        modules=_MODULES,
        header=("import al::* hiding Geo::Inner",),
        probes={
            **_hidden_probes("Geo::Inner"),
            **_hidden_probes("Base::Inner"),
            "routed": rejected("al::Base::Inner(y = 1)", HiddenMemberError, "al::Base::Inner"),
            "alias": accepted("Geo(x = 1)", _RECORD),
            "target": accepted("Base(x = 1)", _RECORD),
            "sibling": accepted("Base::f", "() -> int"),
        },
    ),
    "hiding-an-alias-removes-its-target": Scenario(
        modules=_MODULES,
        header=("import al::* hiding Geo",),
        probes={
            "target-value": rejected("Base(x = 1)", AglScopeError, "Base"),
            "beneath-the-target": rejected("Base::Inner(y = 1)", HiddenMemberError, "Base::Inner"),
            "beneath-the-alias": rejected("Geo::f", HiddenMemberError, "Geo::f"),
            "routed-beneath-the-alias": rejected("al::Geo::f", HiddenMemberError, "al::Geo::f"),
            "anchored-beneath-the-alias": rejected("/al::Geo::f", HiddenMemberError, "/al::Geo::f"),
            "beneath-it-through-another-alias": rejected(
                "fn(p: Sh) => p", AglTypeError, "p: Sh", phase="typecheck"
            ),
            "unrelated-alias": accepted("fn(p: Uh) => p", "base::Oops -> base::Oops"),
        },
    ),
    "an-item-path-through-an-alias-names-a-declaration": Scenario(
        modules=_MODULES,
        header=(),
        probes={
            "hiding": rejected(
                "import al::* hiding Geo::Nope\n1",
                UnknownMemberError,
                "import al::* hiding Geo::Nope",
            ),
            "tail": rejected(
                "import al::{Geo::Nope}\n1", UnknownMemberError, "import al::{Geo::Nope}"
            ),
        },
    ),
    "an-import-tail-names-a-path-through-an-alias": Scenario(
        modules=_MODULES,
        header=("import al::{Geo::Inner as I, Hue::Red, Geo::Shape}",),
        probes={
            "beneath-it": accepted(
                "fn(p: Geo::Shape) => p is Geo::Shape::Sq", "base::Base::Shape -> bool"
            ),
            "beneath-it-value": accepted("Geo::Shape::Sq", "record base::Base::Shape::Sq"),
            "beneath-it-unknown": rejected(
                "Geo::Shape::Nope", UnknownMemberError, "Geo::Shape::Nope"
            ),
            "path-value": accepted("Geo::Inner(y = 1)", _INNER),
            **type_positions("path", "Geo::Inner", "base::Base::Inner"),
            "rename-value": accepted("I(y = 1)", _INNER),
            "rename-pattern": accepted(
                "fn(p: Geo::Inner) => case p of\n  | I(y) => y", "base::Base::Inner -> int"
            ),
            "member": accepted("Hue::Red", "record base::Color::Red"),
            "alias-not-exposed": rejected("Geo(x = 1)", AglScopeError, "Geo"),
            "region": accepted(_in_region("import al::{Geo::Inner as J}", "J(y = 1)"), _INNER),
        },
    ),
    "a-use-hiding-removes-the-declaration-from-every-spelling-it-brings": Scenario(
        modules=_MODULES,
        header=("import al", "import base"),
        probes={
            "through-the-alias": rejected(
                _in_region("use al::* hiding Base::Inner", "Geo::Inner(y = 1)"),
                HiddenMemberError,
                "Geo::Inner",
            ),
            "named-through-an-alias": rejected(
                _in_region("use al::* hiding Geo::Inner", "Base::Inner(y = 1)"),
                HiddenMemberError,
                "Base::Inner",
            ),
            "alias-kept": accepted(
                _in_region("use al::* hiding Geo::Inner", "Geo(x = 1)"), _RECORD
            ),
            "tail-through-an-alias": accepted(
                _in_region("use al::{Geo::Inner}", "Geo::Inner(y = 1)"), _INNER
            ),
            "target-removes-its-alias": rejected(
                _in_region("use al::* hiding Base", "Geo(x = 1)"), AglScopeError, "Geo"
            ),
            "target-removes-its-alias-pattern": rejected(
                _in_region("use al::* hiding Base", _BASE_CASE), NoVisibleConstructorError, "Geo(x)"
            ),
            "target-hides-beneath-its-alias": rejected(
                _in_region("use al::* hiding Base", "Geo::Inner(y = 1)"),
                HiddenMemberError,
                "Geo::Inner",
            ),
            "member-through-an-alias": rejected(
                _in_region("use al::* hiding Hue::Red", "Red"), AglScopeError, "Red"
            ),
            "member-kept": accepted(_in_region("use al::* hiding Hue::Red", "Green"), _GREEN),
        },
    ),
    "a-region-import-hiding-removes-the-declaration": Scenario(
        modules=_MODULES,
        header=("import base",),
        probes={
            "alias-value": rejected(
                _in_region("import al::* hiding Base", "Geo(x = 1)"), AglScopeError, "Geo"
            ),
            "alias-pattern": rejected(
                _in_region("import al::* hiding Base", _BASE_CASE),
                NoVisibleConstructorError,
                "Geo(x)",
            ),
            "beneath-the-alias": rejected(
                _in_region("import al::* hiding Base", "Geo::Inner(y = 1)"),
                HiddenMemberError,
                "Geo::Inner",
            ),
            "alias-kept": accepted(
                _in_region("import al::* hiding Base::Inner", "Geo(x = 1)"), _RECORD
            ),
            "member-through-an-alias": rejected(
                _in_region("import al::* hiding Hue::Red", "Red"), AglScopeError, "Red"
            ),
            "member-kept": accepted(_in_region("import al::* hiding Hue::Red", "Green"), _GREEN),
        },
    ),
    "hiding-an-enum-member-through-an-alias": Scenario(
        modules=_MODULES,
        header=("import al::* hiding Hue::Red",),
        probes={
            "value": rejected("Red", AglScopeError, "Red"),
            "pattern": rejected(
                "fn(p: Color) => case p of\n  | Red => 1\n  | _ => 2",
                NoVisibleConstructorError,
                "Red",
            ),
            "through-the-alias": rejected("Hue::Red", HiddenMemberError, "Hue::Red"),
            "through-the-target": rejected("Color::Red", HiddenMemberError, "Color::Red"),
            "is": rejected("fn(p: Color) => p is Hue::Red", HiddenMemberError, "Hue::Red"),
            "type": rejected("fn(p: Hue::Red) => p", HiddenMemberError, "Hue::Red"),
            "kept": accepted("Green", _GREEN),
        },
    ),
    "an-export-item-names-a-path-through-an-alias": Scenario(
        modules={**_MODULES, "ex": _EXPORTING_THROUGH},
        header=("import ex::*",),
        probes={
            "path-value": accepted("Geo::Inner(y = 1)", _INNER),
            **type_positions("path", "Geo::Inner", "base::Base::Inner"),
            "path-pattern": accepted(
                "fn(p: I) => case p of\n  | Geo::Inner(y) => y", "base::Base::Inner -> int"
            ),
            "rename-value": accepted("I(y = 1)", _INNER),
            "rename-pattern": accepted(
                "fn(p: Geo::Inner) => case p of\n  | I(y) => y", "base::Base::Inner -> int"
            ),
            "nested": accepted(
                "fn(p: Geo::Shape) => p is Geo::Shape::Sq", "base::Base::Shape -> bool"
            ),
            "nested-value": accepted("Geo::Shape::Sq", "record base::Base::Shape::Sq"),
            "function": accepted("Geo::f", "() -> int"),
            "member": accepted("Hue::Red", "record base::Color::Red"),
            "routed": accepted("ex::Geo::Inner(y = 1)", _INNER),
            "routed-rename": accepted("ex::I(y = 1)", _INNER),
            "alias-not-exposed": rejected("Geo(x = 1)", AglScopeError, "Geo"),
            "info": info("I", "I is a constructor.\nSignature:\n  I(y: int) -> base::Base::Inner"),
        },
    ),
    "an-export-hiding-names-a-path-through-an-alias": Scenario(
        modules={**_MODULES, "ex": _HIDING_THROUGH},
        header=("import ex::*",),
        probes={
            **_hidden_probes("Base::Inner"),
            "routed": rejected("ex::Base::Inner(y = 1)", HiddenMemberError, "ex::Base::Inner"),
            "target": accepted("Base(x = 1)", _RECORD),
            "sibling": accepted("Base::f", "() -> int"),
        },
    ),
    "an-alias-spelling-reads-as-its-target-in-every-position": Scenario(
        modules=_MODULES,
        header=("import al::*",),
        probes={
            "receiver": accepted("def Geo::m(self) -> int = self.x\nGeo(x = 2).m()", "int"),
            "receiver-like-its-target": accepted(
                "def Base::m(self) -> int = self.x\nBase(x = 2).m()", "int"
            ),
            "use-glob": accepted(_in_region("use Geo::*", "Inner(y = 1)"), _INNER),
            "is": accepted(
                "fn(p: Base::Shape) => p is Geo::Shape::Sq", "base::Base::Shape -> bool"
            ),
            "pattern": accepted(
                "fn(p: Base::Shape) => case p of\n  | Geo::Shape::Sq => 1\n  | _ => 2",
                "base::Base::Shape -> int",
            ),
            "catch": accepted("try\n  1\ncatch Uh as e =>\n  e.code", "int"),
            "extends": accepted(
                "exception Local extends Uh\nfn(p: Local) => p.code", "Local -> int"
            ),
            "bare-type-entry": rejected(
                "Sh", AglTypeError, "Sh", type_entry="enum base::Base::Shape\n  | Sq\n  | Ci"
            ),
            "generic": accepted("fn(p: Bx[int]) => p.v", "base::Box[int] -> int"),
            "generic-nested": accepted("Bx::In(w = 1)", "record base::Box::In\n  w: int"),
            "generic-nested-takes-no-arguments": rejected(
                "Bx[int]::In(w = 1)", TypeArgumentsError, "Bx[int]"
            ),
        },
    ),
}


class TestTransparentAliases:
    """Alias spellings in every position, file mode and every REPL grouping."""

    @pytest.mark.parametrize("scenario", scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", file_params(_SCENARIOS))
    def test_a_file_resolves_like_the_inline_entry(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        assert_file_resolves_like_inline_entry(tmp_path, scenario)


_UNRESOLVED = "import base\ntype Geo = Base\n"
"""Its alias target names nothing where it is declared: ``base`` is imported qualified."""


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param("import al::*\n1", id="unused"),
        pytest.param("import al::*\nGeo::Inner(y = 1)", id="beneath-it"),
        pytest.param("import al::*\nGeo(x = 1)", id="constructed"),
        pytest.param("import al\nfn(p: al::Geo) => p", id="annotated"),
    ],
)
def test_an_unresolved_alias_target_is_an_error_where_the_alias_is_declared(
    tmp_path: Path, entry: str
) -> None:
    phase, error, span, _identity = file_verdict(
        tmp_path, {"entry": entry, "base": _BASE, "al": _UNRESOLVED}
    )

    assert (phase, error) == ("scope", AglTypeError)
    assert span is not None and span.source.label.endswith("al.agl")
    assert span_text(_UNRESOLVED, span) == "type Geo = Base"


@pytest.mark.parametrize(
    "export",
    [
        pytest.param("import al\nexport al::{Geo::Nope}\n", id="item"),
        pytest.param("import al\nexport al hiding Geo::Nope\n", id="hiding"),
    ],
)
def test_an_export_item_through_an_alias_naming_nothing_is_an_error_where_written(
    tmp_path: Path, export: str
) -> None:
    phase, error, span, _identity = file_verdict(
        tmp_path, {"entry": "import ex\n1", "ex": export, **_MODULES}
    )

    assert (phase, error) == ("scope", UnknownMemberError)
    assert span is not None and span.source.label.endswith("ex.agl")
    assert span_text(export, span) == export.splitlines()[1]
