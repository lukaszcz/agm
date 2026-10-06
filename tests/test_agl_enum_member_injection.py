"""A bare enum member is injected wherever its enum is reached, over every contribution.

An enum reached at a step injects its members' names there: an own enum, an
own alias of an enum, an import tail at the root or in a region, a ``use``,
or a module qualifier. A record or exception constructor of the same name
claims the spelling over the member, and a removed enum or member injects
nothing.

Every probe is checked in the file part and in every legal REPL grouping of its
scenario's header (see :mod:`tests.agl.qualifier_support`).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import pytest

from agm.agl.diagnostics import HiddenMemberError
from agm.agl.scope.symbols import (
    AglScopeError,
    AmbiguousConstructorError,
    NoVisibleConstructorError,
)
from tests.agl.qualifier_support import (
    Probe,
    Scenario,
    accepted,
    all_groupings,
    assert_scenario,
    rejected,
    scenario_params,
)

_COLOR = "enum Color\n  | Red\n  | Blue\n"
_RED_RECORD = "record Red\n  x: int\n"
_RED_EXCEPTION = "exception Red\n"
_OPT = (
    "enum Opt[T]\n  | Nothing\n  | Just(value: T)\n"
    "type IntOpt = Opt[int]\ntype DecOpt = Opt[decimal]\n"
)
_REFERENCED = "import other\n\nenum E = other::Rec | Other\n"
_MODULES = {
    "a": _COLOR,
    "b": _RED_RECORD,
    "x": _RED_EXCEPTION,
    "opt": _OPT,
    "c": _COLOR + _RED_RECORD,
    "d": "enum Light\n  | Red\n",
    "e": "record Blue\n  n: int\n",
    "nothing": "record Nothing\n",
    "other": "record Rec\n  x: int\n",
    "ref": _REFERENCED,
}
_RED = "record a::Color::Red"
_BLUE = "record a::Color::Blue"
_COLOR_TYPE = "enum a::Color\n  | Red\n  | Blue"


def _selections(red: str, blue: str, annotation: str) -> dict[str, Probe]:
    """Pattern and ``is`` probes spelling ``Color``'s members *red* and *blue*."""
    return {
        "pattern": accepted(
            f"let v: {annotation} = {blue}\ncase v of\n  | {red} => 1\n  | _ => 2", "int"
        ),
        "is": accepted(f"let v: {annotation} = {blue}\nv is {red}", "bool"),
    }


def _members(red: str, blue: str, annotation: str, red_identity: str = _RED) -> dict[str, Probe]:
    """Value, pattern and ``is`` probes spelling ``Color``'s members *red* and *blue*."""
    return {"value": accepted(red, red_identity), **_selections(red, blue, annotation)}


def _bare(annotation: str = "Color", identity: str = _RED) -> dict[str, Probe]:
    return _members("Red", "Blue", annotation, identity)


def _record_selections() -> dict[str, Probe]:
    """The pattern probe whose scrutinee is the record ``Red``."""
    return {
        "record-pattern": accepted("let v: Red = Red(x = 1)\ncase v of\n  | Red(x) => x", "int"),
    }


def _same_name(value: Probe, selected: str = "Color") -> dict[str, Probe]:
    """Value probe plus the scrutinee-directed pattern and ``is`` probes of a clash."""
    return {"value": value, **_selections("Red", "Blue", selected), **_record_selections()}


def _in_region(*lines: str, tail: str) -> str:
    """*lines* inside ``scope s`` followed by the statement *tail*."""
    body = "\n".join(f"  {line}" for line in "\n".join(lines).splitlines())
    return f"scope s\n{body}\nend s\n\n{tail}"


def _region_selections(imports: str, annotation: str, red: str = "Red") -> dict[str, Probe]:
    """Pattern and ``is`` probes spelling the members of an enum reached in a region."""
    return {
        "pattern": accepted(
            _in_region(
                imports,
                "def f() -> int =",
                f"  let v: {annotation} = Blue",
                "  case v of",
                f"    | {red} => 1",
                "    | _ => 2",
                tail="s::f()",
            ),
            "int",
        ),
        "is": accepted(
            _in_region(
                imports,
                "def f() -> bool =",
                f"  let v: {annotation} = Blue",
                f"  v is {red}",
                tail="s::f()",
            ),
            "bool",
        ),
    }


def _region_members(imports: str, annotation: str) -> dict[str, Probe]:
    """Value, pattern and ``is`` probes spelling the members of an enum imported in a region."""
    return {
        "value": accepted(_in_region(imports, "let q = Red", tail="s::q"), _RED),
        **_region_selections(imports, annotation),
    }


def _scenario(header: tuple[str, ...], probes: Mapping[str, Probe]) -> Scenario:
    return Scenario(modules=_MODULES, header=header, probes=probes)


_INJECTING = {
    "own-enum": Scenario(header=(_COLOR.rstrip(),), probes=_bare(identity="record Color::Red")),
    "root-tail": _scenario(("import a::Color",), _bare()),
    "root-tail-glob": _scenario(("import a::*",), _bare()),
    "root-tail-rename": _scenario(("import a::{Color as Hue}",), _bare("Hue")),
    "region-tail": _scenario((), _region_members("import a::Color", "Color")),
    "region-tail-rename": _scenario((), _region_members("import a::{Color as Hue}", "Hue")),
    "use-glob": _scenario(("import a", "use a::*"), _bare()),
    "use-item": _scenario(("import a", "use a::Color"), _bare()),
    "use-rename": _scenario(("import a", "use a::{Color as K}"), _bare("K")),
    "use-alias": _scenario(("import a", "use a::Color as K"), _bare("K")),
    "module-qualifier": _scenario(
        ("import a",),
        {
            "route": accepted("a::Red", _RED),
            "anchored-route": accepted("/a::Red", _RED),
        },
    ),
}

_PARAM_DEFAULT = accepted("def f(x: Color = Red) -> Color = x\nf()", _COLOR_TYPE)
_FIELD_DEFAULT = accepted("record Box\n  c: Color = Red\n\nBox().c", _COLOR_TYPE)

_DEFAULTS = {
    "root-tail": _scenario(
        ("import a::Color",), {"param": _PARAM_DEFAULT, "field": _FIELD_DEFAULT}
    ),
    "root-tail-rename": _scenario(
        ("import a::{Color as Hue}",),
        {
            "param": _PARAM_DEFAULT,
            "field": accepted("record Box\n  c: Hue = Red\n\nBox().c", _COLOR_TYPE),
        },
    ),
    "use-glob": _scenario(
        ("import a", "use a::*"), {"param": _PARAM_DEFAULT, "field": _FIELD_DEFAULT}
    ),
    "route": _scenario(
        ("import a",),
        {
            "param": accepted("def f(x: a::Color = a::Red) -> a::Color = x\nf()", _COLOR_TYPE),
            "field": accepted("record Box\n  c: a::Color = a::Red\n\nBox().c", _COLOR_TYPE),
        },
    ),
    "region-tail": _scenario(
        (),
        {
            "param": accepted(
                _in_region("import a::Color", "def f(x: Color = Red) -> Color = x", tail="s::f()"),
                _COLOR_TYPE,
            ),
            "field": accepted(
                _in_region(
                    "import a::Color",
                    "record Box",
                    "  c: Color = Red",
                    tail="s::Box().c",
                ),
                _COLOR_TYPE,
            ),
        },
    ),
}

_REC = "record other::Rec\n  x: int"
_OTHER = "record ref::E::Other"
_NOTHING = "record opt::Opt::Nothing"
_JUST_INT = "record opt::Opt::Just[int]\n  value: int"
_JUST_TEXT = "record opt::Opt::Just[text]\n  value: text"
_RECORD = "record b::Red\n  x: int"
_EXCEPTION = "x::Red"

_MEMBER_KINDS = {
    "inline-member": _scenario(("import ref::E",), _members("Other", "Other", "E", _OTHER)),
    "referenced-member": _scenario(
        ("import ref::E",),
        {
            "value": accepted("Rec(x = 1)", _REC),
            "pattern": accepted("let v: E = Other\ncase v of\n  | Rec(x) => x\n  | _ => 2", "int"),
            "is": accepted("let v: E = Other\nv is Rec", "bool"),
        },
    ),
    "member-through-applying-alias": _scenario(
        ("import opt::IntOpt",),
        {
            "value": accepted("Nothing", _NOTHING),
            "pattern": accepted(
                "let v: IntOpt = IntOpt::Nothing\ncase v of\n  | Nothing => 1\n  | _ => 2", "int"
            ),
            "is": accepted("let v: IntOpt = IntOpt::Nothing\nv is Nothing", "bool"),
        },
    ),
    "generic-member": _scenario(
        ("import opt::Opt",),
        {
            "nothing": accepted("Nothing", _NOTHING),
            "just": accepted("Just(value = 1)", _JUST_INT),
            "let": accepted("let z = Just(value = 1)\nz", _JUST_INT),
        },
    ),
    "scrutinee-selects-among-generic-and-applied": _scenario(
        ("import opt::{Opt, IntOpt}",),
        {
            "pattern": accepted(
                "let v: IntOpt = IntOpt::Nothing\ncase v of\n  | Nothing => 1\n  | _ => 2", "int"
            ),
            "is": accepted("let v: IntOpt = IntOpt::Nothing\nv is Nothing", "bool"),
        },
    ),
}

_REMOVED = {
    "hidden-enum": _scenario(
        ("import a::* hiding Color",),
        {
            "bare": rejected("Red", AglScopeError, "Red"),
        },
    ),
    "hidden-member": _scenario(
        ("import a::* hiding Color::Red",),
        {
            "hidden": rejected("Red", AglScopeError, "Red"),
            "hidden-qualified": rejected("Color::Red", HiddenMemberError, "Color::Red"),
            "kept": accepted("Blue", _BLUE),
            "pattern": rejected(
                "let v: Color = Blue\ncase v of\n  | Red => 1\n  | _ => 2",
                NoVisibleConstructorError,
                "Red",
            ),
        },
    ),
    "hidden-member-use": _scenario(
        ("import a", "use a::* hiding Color::Red"),
        {
            "hidden": rejected("Red", AglScopeError, "Red"),
            "route-fallback": accepted("a::Color::Red", _RED),
            "kept": accepted("Blue", _BLUE),
        },
    ),
    "hidden-member-region-import": _scenario(
        (),
        {
            "hidden": rejected(
                _in_region("import a::* hiding Color::Red", "let q = Red", tail="s::q"),
                AglScopeError,
                "Red",
            ),
            "kept": accepted(
                _in_region("import a::* hiding Color::Red", "let q = Blue", tail="s::q"), _BLUE
            ),
        },
    ),
    "hidden-enum-use": _scenario(
        ("import a", "use a::* hiding Color"),
        {
            "hidden": rejected("Red", AglScopeError, "Red"),
            "route-fallback": accepted("a::Color::Red", _RED),
        },
    ),
}

_HIDDEN_CLASH = {
    "hidden-record": _scenario(("import a::*", "import b::* hiding Red"), _bare()),
    "hidden-record-beside-a-visible-member": _scenario(
        ("import a::*", "import e::* hiding Blue"),
        {"value": accepted("Blue", _BLUE), "pattern": _bare()["pattern"]},
    ),
    "hidden-record-renamed-enum": _scenario(
        ("import a::{Color as Hue}", "import b::* hiding Red"), _bare("Hue")
    ),
    "hidden-record-applied-alias": _scenario(
        ("import opt::{IntOpt}", "import nothing::* hiding Nothing"),
        {"value": accepted("Nothing", _NOTHING)},
    ),
    "hidden-record-region": _scenario(
        (), _region_members("import a::Color\nimport b::* hiding Red", "Color")
    ),
}

_ALIASES = {
    "own-alias-qualified": _scenario(
        ("import a", "type C = a::Color"), {"value": accepted("C::Red", _RED)}
    ),
    "own-applied-alias-qualified": _scenario(
        ("import opt", "type IO = opt::Opt[int]"), {"value": accepted("IO::Nothing", _NOTHING)}
    ),
}

_ROUTES = {
    "own-root": Scenario(
        header=(_COLOR.rstrip(),), probes={"value": accepted("::Red", "record Color::Red")}
    ),
    "referenced-via-route": _scenario(
        ("import ref",),
        {
            "inline": accepted("ref::Other", _OTHER),
            "anchored": accepted("/ref::Other", _OTHER),
        },
    ),
}


class _Clash(NamedTuple):
    """A same-named enum member and record or exception reached through some contributions.

    *red* names the probes that clash and *by_use* says whether they fail because
    a ``use`` does not yield (otherwise because the scrutinee does not select).
    """

    header: tuple[str, ...]
    probes: dict[str, Probe]
    red: frozenset[str]
    by_use: bool


def _clash(
    header: tuple[str, ...],
    value: Probe,
    red: tuple[str, ...],
    *,
    by_use: bool = False,
    exception: bool = False,
    region: str | None = None,
) -> _Clash:
    """Value probe plus the scrutinee-directed probes for *header*'s ``Color`` and ``Red``."""
    if region is not None:
        probes = {"value": value, **_region_selections(region, "Color")}
    elif exception:
        probes = {"value": value, **_selections("Red", "Blue", "Color")}
    else:
        probes = {"value": value, **_selections("Red", "Blue", "Color"), **_record_selections()}
    return _Clash(header, probes, frozenset(red), by_use)


_RECORD_VALUE = accepted("Red(x = 1)", _RECORD)
_EXCEPTION_VALUE = accepted('Red(message = "m")', _EXCEPTION)
_C_RECORD_VALUE = accepted("Red(x = 1)", "record c::Red\n  x: int")
_USE_RED = ("value", "record-pattern")
_SCRUTINEE_RED = ("pattern", "is")
_REGION_RECORD = _in_region("import a::Color", "import b::Red", "let q = Red(x = 1)", tail="s::q")
_REGION_USE = _in_region(
    "import a::Color", "import b", "use b::Red", "let q = Red(x = 1)", tail="s::q"
)

_CLASHES = {
    "own-enum-and-record": _clash(
        (_COLOR.rstrip(), _RED_RECORD.rstrip()),
        accepted("Red(x = 1)", "record Red\n  x: int"),
        ("pattern", "is"),
    ),
    "two-imports": _clash(("import a::Color", "import b::Red"), _RECORD_VALUE, _SCRUTINEE_RED),
    "two-imports-exception": _clash(
        ("import a::Color", "import x::Red"),
        _EXCEPTION_VALUE,
        ("pattern", "is"),
        exception=True,
    ),
    "one-import-tail": _clash(("import c::*",), _C_RECORD_VALUE, _SCRUTINEE_RED),
    "one-use": _clash(("import c", "use c::*"), _C_RECORD_VALUE, _SCRUTINEE_RED),
    "glob-import-and-glob-use": _clash(
        ("import c::*\nimport c", "use c::*"), _C_RECORD_VALUE, _SCRUTINEE_RED
    ),
    "region-two-imports": _clash(
        (),
        accepted(_REGION_RECORD, _RECORD),
        ("pattern", "is"),
        region="import a::Color\nimport b::Red",
    ),
    "use-and-use": _clash(
        ("import a\nimport b", "use a::Color", "use b::Red"),
        _RECORD_VALUE,
        _USE_RED,
        by_use=True,
    ),
    "use-and-use-exception": _clash(
        ("import a\nimport x", "use a::Color", "use x::Red"),
        _EXCEPTION_VALUE,
        ("value",),
        by_use=True,
        exception=True,
    ),
    "import-enum-use-record": _clash(
        ("import a::Color\nimport b", "use b::Red"), _RECORD_VALUE, _USE_RED, by_use=True
    ),
    "use-enum-import-record": _clash(
        ("import a\nimport b::Red", "use a::Color"), _RECORD_VALUE, _USE_RED, by_use=True
    ),
    "region-import-enum-use-record": _clash(
        (),
        accepted(_REGION_USE, _RECORD),
        ("value",),
        by_use=True,
        region="import a::Color\nimport b\nuse b::Red",
    ),
}


def _clash_table(*, red: bool | None = None, by_use: bool = False) -> dict[str, Scenario]:
    """The clash rows' passing probes (``red`` false) or their failing ones (``red`` true)."""
    table: dict[str, Scenario] = {}
    for name, clash in _CLASHES.items():
        if red and clash.by_use != by_use:
            continue
        probes = {k: v for k, v in clash.probes.items() if (k in clash.red) == bool(red)}
        if probes:
            table[name] = _scenario(clash.header, probes)
    return table


_HIDDEN_QUALIFIED = {
    "hidden-member-qualified": _scenario(
        ("import a hiding Color::Red",),
        {
            "route": rejected("a::Red", HiddenMemberError, "a::Red"),
            "anchored-route": rejected("/a::Red", HiddenMemberError, "/a::Red"),
        },
    ),
    "hidden-enum-qualified": _scenario(
        ("import a hiding Color",),
        {
            "route": rejected("a::Red", HiddenMemberError, "a::Red"),
            "anchored-route": rejected("/a::Red", HiddenMemberError, "/a::Red"),
        },
    ),
}


def _withheld_scenario(use: str) -> Scenario:
    """``mid`` re-exports ``a`` withholding ``Color::Red``; bare ``Red`` stays unknown."""
    return Scenario(
        modules={**_MODULES, "mid": "import a\nexport a hiding Color::Red\n"},
        header=("import mid", use),
        probes={"bare": rejected("Red", AglScopeError, "Red")},
    )


_USE_IGNORES_WAYS = {
    "use-glob-of-an-import-hiding-a-member": _scenario(
        ("import a hiding Color::Red", "use a::*"),
        {"bare": rejected("Red", AglScopeError, "Red")},
    ),
    "use-item-of-an-import-hiding-a-member": _scenario(
        ("import a hiding Color::Red", "use a::{Color}"),
        {"bare": rejected("Red", AglScopeError, "Red")},
    ),
    "use-glob-of-an-export-withholding-a-member": _withheld_scenario("use mid::*"),
    "use-item-of-an-export-withholding-a-member": _withheld_scenario("use mid::{Color}"),
}

_SAME_NAME = _clash_table()
_SCRUTINEE_SELECTS = _clash_table(red=True)
_USE_YIELDS = _clash_table(red=True, by_use=True)

_ONE_MEMBER = {
    "applied-and-generic-value": _scenario(
        ("import opt::{Opt, IntOpt}",), {"value": accepted("Nothing", _NOTHING)}
    ),
    "applied-and-generic-constructor": _scenario(
        ("import opt::{Opt, IntOpt}",), {"value": accepted("Just(value = 1)", _JUST_INT)}
    ),
}

_GENERIC_MEMBER = {
    "lone-applied-alias-other-argument": _scenario(
        ("import opt::{IntOpt}",), {"value": accepted('Just(value = "x")', _JUST_TEXT)}
    ),
    "lone-applied-alias-inferred": _scenario(
        ("import opt::{DecOpt}",), {"value": accepted("let z = Just(value = 1)\nz", _JUST_INT)}
    ),
}

_OWN_ALIAS_CLASH = {
    "alias-beside-other-enum": _scenario(
        ("import a\nimport d::*", "type C = a::Color"),
        {"value": rejected("Red", AmbiguousConstructorError, "Red")},
    )
}

_OWN_ALIAS_REGION = {
    "alias-in-region": _scenario(
        (),
        {
            "value": accepted(
                _in_region("import a", "type C = a::Color", "let q = Red", tail="s::q"), _RED
            )
        },
    )
}

_OWN_ALIAS = {
    "renaming-alias": _scenario(("import a", "type C = a::Color"), _bare("C")),
    "applying-alias": _scenario(
        ("import opt", "type IO = opt::Opt[int]"), {"value": accepted("Nothing", _NOTHING)}
    ),
}


def _use_with_own_scope(header: tuple[str, ...], probes: Mapping[str, Probe]) -> Scenario:
    """A scenario whose ``use`` of the own scope must share its REPL entry with the scope."""
    use = next(i for i, line in enumerate(header) if line.startswith("use "))
    own = len(header) - 1
    legal = frozenset(
        sizes
        for sizes in all_groupings(len(header) + 1)
        if any(sum(sizes[:k]) <= use and own < sum(sizes[: k + 1]) for k in range(len(sizes)))
    )
    return Scenario(modules=_MODULES, header=header, probes=probes, legal=legal)


_OWN_HUE = "scope S\n  enum Hue\n    | Red\n    | Green\nend S"
_HUE_RED = "record S::Hue::Red"
_RENAMED_MODULES = {**_MODULES, "mid": "import a\nexport a::{Color as Hue}\n"}
_HIDING_MODULES = {**_MODULES, "mid": "import a\nexport a hiding Color::Red\n"}

_OWN_MEMBER_LAYER = {
    "use-item-beside-imported-record": _use_with_own_scope(
        ("import b::*", "use S::Hue", _OWN_HUE), {"value": accepted("Red", _HUE_RED)}
    ),
    "use-glob-beside-imported-record": _use_with_own_scope(
        ("import b::*", "use S::*", _OWN_HUE), {"value": accepted("Red", _HUE_RED)}
    ),
    "use-item-beside-imported-member": _use_with_own_scope(
        ("import a::*", "use S::Hue", _OWN_HUE), {"value": accepted("Red", _HUE_RED)}
    ),
    "use-glob-beside-imported-member": _use_with_own_scope(
        ("import a::*", "use S::*", _OWN_HUE), {"value": accepted("Red", _HUE_RED)}
    ),
    "region-use-beside-region-import": _scenario(
        (
            _OWN_HUE,
            "scope R\n  import b::*\n  use ::S::Hue\n  let v = Red\nend R",
        ),
        {"value": accepted("R::v", _HUE_RED)},
    ),
    "root-alias-beside-imported-record": _scenario(
        ("import b::*", _OWN_HUE, "type H = S::Hue"), {"value": accepted("Red", _HUE_RED)}
    ),
    "imported-enum-by-use-yields-to-imported-record": _scenario(
        ("import b::*", "import a", "use a::Color"), {"value": accepted("Red(x = 1)", _RECORD)}
    ),
}

_REEXPORTED_MEMBERS = {
    "renamed-enum-by-use": Scenario(
        modules=_RENAMED_MODULES,
        header=("import mid", "use mid::*"),
        probes={"value": accepted("Red", _RED)},
    ),
    "renamed-enum-region-import": Scenario(
        modules=_RENAMED_MODULES,
        header=(),
        probes={"value": accepted(_in_region("import mid::*", "let q = Red", tail="s::q"), _RED)},
    ),
    "withheld-member-qualified": Scenario(
        modules=_HIDING_MODULES,
        header=("import mid",),
        probes={"route": rejected("mid::Red", HiddenMemberError, "mid::Red")},
    ),
    "withheld-member-by-use": Scenario(
        modules=_HIDING_MODULES,
        header=("import mid", "use mid::*"),
        probes={
            "bare": rejected("Red", AglScopeError, "Red"),
            "is": rejected("mid::Blue is Red", NoVisibleConstructorError, "mid::Blue is Red"),
        },
    ),
}


class TestInjection:
    """Every contribution that reaches an enum injects its members bare."""

    @pytest.mark.parametrize("scenario", scenario_params(_INJECTING))
    def test_file_and_every_repl_grouping_agree(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_MEMBER_KINDS))
    def test_member_kinds(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_REMOVED))
    def test_removal(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_HIDDEN_CLASH))
    def test_hidden_record_does_not_claim_the_bare_member(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_SAME_NAME))
    def test_same_named_record_or_exception(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_ALIASES))
    def test_own_alias_qualified(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_DEFAULTS))
    def test_default_expressions(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_ROUTES))
    def test_module_qualifier(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)


class TestInjectionRules:
    """Behaviour of injection that holds over every contribution alike.

    Yielding to a same-named record or exception is decided at the step, in value
    position only; a bare member is the generic member declaration; own aliases inject.
    """

    @pytest.mark.parametrize("scenario", scenario_params(_USE_YIELDS))
    def test_use_yields_to_record_at_the_step(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_SCRUTINEE_SELECTS))
    def test_scrutinee_selects_member_over_same_named_record(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_HIDDEN_QUALIFIED))
    def test_qualified_spelling_of_removed_member_is_hidden(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_USE_IGNORES_WAYS))
    def test_use_injects_only_members_its_imports_keep(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_ONE_MEMBER))
    def test_applied_alias_and_generic_are_one_member(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_GENERIC_MEMBER))
    def test_lone_applied_alias_injects_generic_member(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_OWN_ALIAS))
    def test_own_alias_injects_members(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_OWN_ALIAS_CLASH))
    def test_own_alias_member_clashes_with_other_enum(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_OWN_ALIAS_REGION))
    def test_own_alias_in_region_injects_members(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_OWN_MEMBER_LAYER))
    def test_own_enum_member_is_own_however_reached(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_REEXPORTED_MEMBERS))
    def test_members_through_a_reexporting_module(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)
