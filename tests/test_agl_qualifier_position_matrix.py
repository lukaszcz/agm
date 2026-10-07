"""Scope decides a qualified name once, identically in every position.

Each qualifier is probed in the value, pattern, ``is`` and cast positions
that take constructors, and in every type position (annotation, alias
target, type argument, cast target, applied type) through the shared
:func:`~tests.agl.qualifier_support.type_positions` templates, in the file part
and in every legal REPL grouping of its scenario's header (see
:mod:`tests.agl.qualifier_support`). An accepted probe's identity renders
the declaration it selected: a type position's probe returns its own
parameter, so a wrong selection of the same kind renders another type.

The ``is`` position's identity is only ``bool``; a ``v is Q`` whose ``Q``
selects a member of another type than ``v``'s is a type mismatch, which the
matrix's "other" outcome pins. The chain-length sweep reads a chain of
length two to five through a local region, an own declaration, a module
route and both anchors.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import ReferencedMemberError
from agm.agl.scope.symbols import (
    AmbiguousQualificationError,
    UnknownMemberError,
    UnknownQualifierError,
)
from agm.agl.typecheck import AglTypeError
from agm.agl.typecheck.checker import EnumOwnerMismatchError
from tests.agl.qualifier_support import (
    Probe,
    Scenario,
    accepted,
    all_groupings,
    assert_scenario,
    option_identity,
    rejected,
    scenario_params,
    type_positions,
    type_positions_rejected,
)

# ---------------------------------------------------------------------------
# A unique, another type's ("other"), an ambiguous and a missing member,
# across five owner spellings.
# ---------------------------------------------------------------------------

_RT = {
    "one/types": "enum Color\n  | Red\n  | Green\n",
    "two/types": "enum Color\n  | Red\n  | Blue\n",
}
_MN = {"m": "enum Color\n  | Red\n  | Green\n", "n": "enum Color\n  | Red\n  | Blue\n"}
_GB = {
    "gm": "enum Box[T]\n  | Full(value: T)\n  | Both\n  | Empty\n",
    "gn": "enum Box[T]\n  | Loaded(value: T)\n  | Both\n  | Other\n",
}
_FACADE = {
    "pkg/a": "enum Thing\n  | Red\n  | Blue\n",
    "pkg/b": "enum Thing\n  | Red\n  | Yellow\n",
}
_IDENTICAL_OWNERS = {"one/owner": "enum Owner = A | B\n", "two/owner": "enum Owner = A | B\n"}
_LOCAL_REGIONS = ("scope a\n  enum E = X | Y\nend a", "scope b\n  enum E = X | Z\nend b")


def _uses_first_legal(total: int, prefix: int) -> frozenset[tuple[int, ...]]:
    """Groupings of *total* items whose first entry holds the leading *prefix* items.

    ``use`` declarations naming scope regions declared later in the header
    resolve only in the entry declaring those regions.
    """
    return frozenset(
        (first, *rest)
        for first in range(prefix, total + 1)
        for rest in all_groupings(total - first)
    )


# name -> (modules, header, owner spelling, {outcome: member}, legal groupings)
_FORMS: dict[
    str,
    tuple[dict[str, str], tuple[str, ...], str, dict[str, str], frozenset[tuple[int, ...]] | None],
] = {
    "route-owner": (
        _RT,
        (
            "import one/types",
            "import two/types",
            "let v: one/types::Color = one/types::Color::Green",
        ),
        "types::Color",
        {"unique": "Green", "other": "Blue", "amb": "Red", "missing": "NoSuch"},
        None,
    ),
    "bare-owner": (
        _MN,
        ("import m", "import m::*", "import n::*", "let v: m::Color = m::Color::Green"),
        "Color",
        {"unique": "Green", "other": "Blue", "amb": "Red", "missing": "NoSuch"},
        None,
    ),
    "applied-owner": (
        _GB,
        ("import gm", "import gm::*", "import gn::*", "let v: gm::Box[int] = gm::Box[int]::Empty"),
        "Box[int]",
        {"unique": "Full", "other": "Other", "amb": "Both", "missing": "NoSuch"},
        None,
    ),
    "facade-owner": (
        _FACADE,
        # One entry: a later REPL entry importing ``pkg/a`` replaces the facade's import of it.
        ("import pkg/* as Facade\nimport pkg/a", "let v: pkg/a::Thing = pkg/a::Thing::Blue"),
        "Facade::Thing",
        {"unique": "Blue", "other": "Yellow", "amb": "Red", "missing": "NoSuch"},
        None,
    ),
    "local-use-owner": (
        {},
        ("use a::*", "use b::*", *_LOCAL_REGIONS, "let v: a::E = a::E::Y"),
        "E",
        {"unique": "Y", "other": "Z", "amb": "X", "missing": "NoSuch"},
        _uses_first_legal(6, 4),
    ),
    "module-use-owner": (
        _MN,
        ("import m", "import n", "use m::*", "use n::*", "let v: m::Color = m::Color::Green"),
        "Color",
        {"unique": "Green", "other": "Blue", "amb": "Red", "missing": "NoSuch"},
        None,
    ),
}

# Forms whose applied owner two imports declare: its type arguments apply to no one type.
_AMBIGUOUS_APPLIED_OWNERS = frozenset({"applied-owner"})

# The member type each accepted (form, outcome) selects.
_CASE_TYPE: dict[tuple[str, str], str] = {
    ("route-owner", "unique"): "one/types::Color::Green",
    ("route-owner", "other"): "two/types::Color::Blue",
    ("bare-owner", "unique"): "m::Color::Green",
    ("bare-owner", "other"): "n::Color::Blue",
    ("facade-owner", "unique"): "pkg/a::Thing::Blue",
    ("facade-owner", "other"): "pkg/b::Thing::Yellow",
    ("local-use-owner", "unique"): "a::E::Y",
    ("local-use-owner", "other"): "b::E::Z",
    ("module-use-owner", "unique"): "m::Color::Green",
    ("module-use-owner", "other"): "n::Color::Blue",
}

# The owner enum ``v`` is, which a pattern's bound value renders as.
_OWNER_ENUM: dict[str, str] = {
    "route-owner": "enum one/types::Color\n  | Red\n  | Green",
    "bare-owner": "enum m::Color\n  | Red\n  | Green",
    "facade-owner": "enum pkg/a::Thing\n  | Red\n  | Blue",
    "local-use-owner": "enum a::E\n  | X\n  | Y",
    "module-use-owner": "enum m::Color\n  | Red\n  | Green",
}

# The positions testing ``v`` against a member: each needs ``v``'s own type.
_SCRUTINEE_POSITIONS: dict[str, str] = {
    "pattern": "case v of\n  | {q} as bound => bound\n  | _ => v",
    "is": "v is {q}",
    "narrow": "v as? {q}",
}


def _form_probes(form: str) -> dict[str, Probe]:
    _, _, owner, members, _ = _FORMS[form]
    probes: dict[str, Probe] = {}
    for outcome, member in members.items():
        q = f"{owner}::{member}"
        if outcome in ("amb", "missing") or form in _AMBIGUOUS_APPLIED_OWNERS:
            error = UnknownMemberError if outcome == "missing" else AmbiguousQualificationError
            span = owner if outcome in ("unique", "other") else q
            probes |= type_positions_rejected(outcome, q, error, span)
            probes[f"{outcome}-value"] = rejected(q, error, span)
            for position, text in _SCRUTINEE_POSITIONS.items():
                probes[f"{outcome}-{position}"] = rejected(text.format(q=q), error, span)
            continue
        case_type = _CASE_TYPE[(form, outcome)]
        probes |= type_positions(outcome, q, case_type)
        probes[f"{outcome}-value"] = accepted(q, f"record {case_type}")
        if outcome == "unique":
            probes |= {
                "unique-pattern": accepted(
                    _SCRUTINEE_POSITIONS["pattern"].format(q=q), _OWNER_ENUM[form]
                ),
                "unique-is": accepted(_SCRUTINEE_POSITIONS["is"].format(q=q), "bool"),
                "unique-narrow": accepted(
                    _SCRUTINEE_POSITIONS["narrow"].format(q=q),
                    option_identity(case_type),
                ),
            }
            continue
        # Another type's member: a legal type, but no value ``v`` can be.
        probes |= {
            "other-pattern": rejected(
                _SCRUTINEE_POSITIONS["pattern"].format(q=q),
                EnumOwnerMismatchError,
                q,
                phase="typecheck",
            ),
            "other-is": rejected(
                f"v is {q}", EnumOwnerMismatchError, f"v is {q}", phase="typecheck"
            ),
            "other-narrow": rejected(f"v as? {q}", AglTypeError, f"v as? {q}", phase="typecheck"),
        }
    return probes


def _form_scenario(form: str) -> Scenario:
    modules, header, _, _, legal = _FORMS[form]
    return Scenario(
        modules=modules,
        header=header,
        probes=_form_probes(form),
        legal="ALL" if legal is None else legal,
    )


# ---------------------------------------------------------------------------
# A local scope region never hides a same-named imported region's members.
# ---------------------------------------------------------------------------

_GEO_IMPORTED = "scope Geo\n  record Point\n    x: int\nend Geo\n"


def _imported_point_probes() -> dict[str, Probe]:
    return {
        "point-value": accepted("Geo::Point(x = 1)", "record shapes::Geo::Point\n  x: int"),
        "point-pattern": accepted(
            "fn(p: Geo::Point) => case p of\n  | Geo::Point(x) as w => w",
            "shapes::Geo::Point -> shapes::Geo::Point",
        ),
        **type_positions("point", "Geo::Point", "shapes::Geo::Point"),
    }


# ---------------------------------------------------------------------------
# An own declaration beats two imports of its path, in every position.
# ---------------------------------------------------------------------------

_TWO_IMPORT_LIB = {
    "m": "record Point\n  x: int\nenum Shape\n  | Circle\n",
    "n": "record Point\n  y: int\nenum Shape\n  | Square\n",
}

_LOCAL_WINS_LIB = (
    "scope Geo\n"
    "  record Point\n"
    "    x: int\n"
    "  enum Shape\n"
    "    | Circle\n"
    "    | Square\n"
    "  type Num = int\n"
    "end Geo\n"
)
_LOCAL_WINS_LOCAL = (
    "scope Geo\n  record Point\n    y: int\n  enum Shape\n    | Tri\n  type Num = text\nend Geo"
)
_LOCAL_WINS_NESTED_LIB = (
    "scope Geo\n"
    "  record Point\n"
    "    x: int\n"
    "  record Box[T]\n"
    "    v: T\n"
    "  enum Shape\n"
    "    | Circle\n"
    "    | Square\n"
    "end Geo\n"
)
_LOCAL_WINS_NESTED_LOCAL = "scope Geo\n  record Point\n    y: int\n  enum Shape\n    | Tri\nend Geo"

# ---------------------------------------------------------------------------
# A type-only spelling typed alone at the REPL is a type entry; in a file it
# names no value.
# ---------------------------------------------------------------------------

_TYPE_ENTRY_MODULES = {
    "geo": "type Num = int\n",
    "gm": _GB["gm"],
}

# ---------------------------------------------------------------------------
# Def-created paths beside imported types and regions.
# ---------------------------------------------------------------------------

_DEFPATH_TYPE_OWNER_LIB = "record Geo\n  x: int\nrecord Geo::Inner\n  y: int\n"
_DEFPATH_LIB = "scope Geo\n  record Point\n    x: int\nend Geo\n"
_DEFPATH3_LIB = "scope Geo\n\n  scope In\n    record Point\n      x: int\n  end In\nend Geo\n"
_USED_REGION_LIB = "scope Geo\n  record Point\n    x: int\nend Geo\n"
_USED_REGION_MODULES = {"shapes": _USED_REGION_LIB, "tl": _DEFPATH_TYPE_OWNER_LIB}
_USED_REGION_HEADER = ("import tl::*", "import shapes", "use shapes::*")


# ---------------------------------------------------------------------------
# A chain is looked up as its whole path at every length.
# ---------------------------------------------------------------------------

# ``Geo`` declares a member at every chain length: ``Geo::Point`` (2),
# ``Geo::Shape::Circle`` (3) and ``Geo::Deep::Kind::Round`` (4).
_CHAIN_GEO = (
    "scope Geo\n  record Point\n    x: int\n  enum Shape = Circle | Square\n"
    "\n  scope Deep\n    enum Kind = Round | Flat\n  end Deep\nend Geo"
)
_CHAIN_LOCAL_DEEP = "scope Geo\n\n  scope Deep\n    enum Other = O\n  end Deep\nend Geo"
_CHAIN_NESTED_LOCAL = (
    "scope Geo\n\n  scope Shape\n    enum Kind = Round | Flat\n  end Shape\nend Geo"
)
# Each chain's path below the qualifier, and for a member, its enum's path.
_CHAIN_PATHS: dict[int, tuple[str, str | None]] = {
    2: ("Geo::Point", None),
    3: ("Geo::Shape::Circle", "Geo::Shape"),
    4: ("Geo::Deep::Kind::Round", "Geo::Deep::Kind"),
}


def _chain_probes(qualifier: str, selected: str) -> dict[str, Probe]:
    """Every position's probe of each ``_CHAIN_PATHS`` path written after *qualifier*.

    *selected* prefixes the path the declaration renders as.
    """
    probes: dict[str, Probe] = {}
    for length, (path, owner) in _CHAIN_PATHS.items():
        q = f"{qualifier}{path}"
        s = f"{selected}{path}"
        key = f"length{length}"
        probes |= type_positions(key, q, s)
        if owner is None:
            probes[f"{key}-value"] = accepted(f"{q}(x = 1)", f"record {s}\n  x: int")
            probes[f"{key}-pattern"] = accepted(
                f"fn(p: {q}) => case p of\n  | {q}(x) as w => w", f"{s} -> {s}"
            )
            continue
        o = f"{qualifier}{owner}"
        so = f"{selected}{owner}"
        probes |= {
            f"{key}-value": accepted(q, f"record {s}"),
            f"{key}-pattern": accepted(
                f"fn(p: {o}) => case p of\n  | {q} as c => c\n  | _ => p", f"{so} -> {so}"
            ),
            f"{key}-is": accepted(f"fn(p: {o}) => p is {q}", f"{so} -> bool"),
            f"{key}-narrow": accepted(
                f"fn(p: {o}) => p as? {q}", f"{so} -> std/option::Option[{s}]"
            ),
        }
    return probes


# ---------------------------------------------------------------------------
# A length-4 local scope matching an import route at its leading segment
# alone: the own path wins where it declares the full chain, and the route
# supplies it where the own scope's leaf differs.
# ---------------------------------------------------------------------------

_LEN4_LIB = "scope E\n\n  scope A\n    record Z\n      v: int\n  end A\nend E\n"
_LEN4_LOCAL_SAME_LEAF = (
    "scope Foo\n\n  scope E\n\n    scope A\n      record Z\n        w: int\n    end A\n  end E\n"
    "end Foo"
)
_LEN4_LOCAL_DIFFERENT_LEAF = (
    "scope Foo\n\n  scope E\n\n    scope A\n      record Q\n        w: int\n    end A\n  end E\n"
    "end Foo"
)

_SCENARIOS: dict[str, Scenario] = {
    **{form: _form_scenario(form) for form in _FORMS},
    "facade-owner-alone": Scenario(
        modules=_FACADE,
        header=("import pkg/* as Facade",),
        probes={
            "owner-value": rejected("Facade::Thing", AmbiguousQualificationError, "Facade::Thing"),
            **type_positions_rejected("owner", "Facade::Thing", AmbiguousQualificationError),
        },
    ),
    "applied-owner-identical-in-two-imports": Scenario(
        modules=_IDENTICAL_OWNERS,
        header=("import one/owner::*", "import two/owner::*"),
        probes={
            "amb-value": rejected("Owner[int]::A", AmbiguousQualificationError, "Owner[int]::A"),
            **type_positions_rejected("amb", "Owner[int]::A", AmbiguousQualificationError),
            "missing-value": rejected(
                "Owner[int]::NoSuch", UnknownMemberError, "Owner[int]::NoSuch"
            ),
            "missing-is": rejected(
                "1 is Owner[int]::NoSuch", UnknownMemberError, "Owner[int]::NoSuch"
            ),
            "missing-pattern": rejected(
                "case 1 of\n  | Owner[int]::NoSuch => 1\n  | _ => 2",
                UnknownMemberError,
                "Owner[int]::NoSuch",
            ),
            **type_positions_rejected("missing", "Owner[int]::NoSuch", UnknownMemberError),
        },
    ),
    "local-region-beside-same-named-imported-region": Scenario(
        modules={"shapes": _GEO_IMPORTED},
        header=("import shapes::*", "scope Geo\n  enum Kind = Round | Flat\nend Geo"),
        probes={
            **_imported_point_probes(),
            **type_positions("kind", "Geo::Kind", "Geo::Kind"),
        },
    ),
    "imported-region-member": Scenario(
        modules={"shapes": _GEO_IMPORTED},
        header=("import shapes::*",),
        probes=_imported_point_probes(),
    ),
    "own-types-beside-two-imports": Scenario(
        modules=_TWO_IMPORT_LIB,
        header=("import m::*", "import n::*", "record Point\n  z: int", "enum Shape\n  | Tri"),
        probes={
            "point-value": accepted("Point(z = 1)", "record Point\n  z: int"),
            **type_positions("point", "Point", "Point"),
            "tri-value": accepted("Shape::Tri", "record Shape::Tri"),
            **type_positions("shape", "Shape", "Shape"),
        },
    ),
    "own-region-types-beside-imported-region": Scenario(
        modules={"shapes": _LOCAL_WINS_LIB},
        header=("import shapes::*", _LOCAL_WINS_LOCAL),
        probes={
            **type_positions("shape", "Geo::Shape", "Geo::Shape"),
            **type_positions("num", "Geo::Num", "text"),
            "point-value": accepted("Geo::Point(y = 1)", "record Geo::Point\n  y: int"),
        },
    ),
    "own-region-beside-imported-generic": Scenario(
        modules={"shapes": _LOCAL_WINS_NESTED_LIB},
        header=("import shapes::*", _LOCAL_WINS_NESTED_LOCAL),
        probes={
            **type_positions("shape", "Geo::Shape", "Geo::Shape"),
            **type_positions("box", "Geo::Box[int]", "shapes::Geo::Box[int]"),
        },
    ),
    "type-entries": Scenario(
        modules=_TYPE_ENTRY_MODULES,
        header=(
            "import geo",
            "import gm",
            "scope Geo\n\n  scope In\n    type Num = text\n  end In\nend Geo",
        ),
        probes={
            "module-alias": rejected("geo::Num", AglTypeError, "geo::Num", type_entry="int"),
            "applied-generic": rejected(
                "gm::Box[int]",
                AglTypeError,
                "gm::Box",
                type_entry="enum gm::Box[int]\n  | Full(value: int)\n  | Both\n  | Empty",
            ),
            "nested-alias": rejected(
                "Geo::In::Num", AglTypeError, "Geo::In::Num", type_entry="text"
            ),
        },
    ),
    "generic-alias-owner-nested-in-a-generic-target": Scenario(
        header=(
            "enum Slot[T]\n  | Filled(value: T)\n  | Empty",
            "type Rows[A] = Slot[array[A]]",
            "let row: Slot[array[int]] = Slot::Filled(value = [1, 2])",
        ),
        probes={
            "pattern": accepted(
                "case row of\n  | Rows[int]::Filled(value) => value\n  | Slot::Empty => []",
                "array[int]",
            ),
            **type_positions("member", "Rows[int]::Filled", "Slot::Filled[array[int]]"),
        },
    ),
    "unknown-route": Scenario(
        header=(),
        probes={
            "value": rejected("missing::Item(x = 1)", UnknownQualifierError, "missing::Item"),
            "pattern": rejected(
                "let v = 1\ncase v of\n  | missing::Item => 1\n  | _ => 2",
                UnknownQualifierError,
                "missing::Item",
            ),
            **type_positions_rejected(
                "applied", "missing::Item[int]", UnknownQualifierError, "missing::Item"
            ),
        },
    ),
    "own-enum-beside-imported-enum": Scenario(
        modules={"lib": "record Point\n  x: int\nenum Shape\n  | Circle\n"},
        header=("import lib::*", "enum Shape\n  | Tri\n"),
        probes={
            "circle-value": accepted("Shape::Circle", "record lib::Shape::Circle"),
            "circle-pattern": accepted(
                "fn(p: lib::Shape) => case p of\n  | Shape::Circle as c => c",
                "lib::Shape -> lib::Shape",
            ),
            "circle-is": accepted("fn(p: lib::Shape) => p is Shape::Circle", "lib::Shape -> bool"),
            "circle-narrow": accepted(
                "fn(p: lib::Shape) => p as? Shape::Circle",
                "lib::Shape -> std/option::Option[lib::Shape::Circle]",
            ),
            **type_positions("circle", "Shape::Circle", "lib::Shape::Circle"),
            "tri-value": accepted("Shape::Tri", "record Shape::Tri"),
            **type_positions("shape", "Shape", "Shape"),
        },
    ),
    "def-path-over-imported-type": Scenario(
        modules={"shapes": _DEFPATH_TYPE_OWNER_LIB},
        header=("import shapes::*", "def Geo::m(self) -> int = self.x"),
        probes={
            "inner-value": accepted("Geo::Inner(y = 1)", "record shapes::Geo::Inner\n  y: int"),
            **type_positions("inner", "Geo::Inner", "shapes::Geo::Inner"),
            "nope-value": rejected("Geo::Nope(y = 1)", UnknownMemberError, "Geo::Nope"),
            **type_positions_rejected("nope", "Geo::Nope", UnknownMemberError),
            # ``::`` reads the own root, where ``Geo`` holds only ``m``.
            "anchored-inner-value": rejected(
                "::Geo::Inner(y = 1)", UnknownMemberError, "::Geo::Inner"
            ),
            **type_positions_rejected("anchored-inner", "::Geo::Inner", UnknownMemberError),
            "anchored-nope-value": rejected(
                "::Geo::Nope(y = 1)", UnknownMemberError, "::Geo::Nope"
            ),
        },
    ),
    "referenced-member-walked-past": Scenario(
        header=("record Saved\n  id: int", "enum Stored = ::Saved | Fresh"),
        probes={
            "value": rejected("Stored::Saved::X", ReferencedMemberError, "Stored::Saved::X"),
            **type_positions_rejected("member", "Stored::Saved::X", ReferencedMemberError),
        },
    ),
    "use-region-beside-imported-type": Scenario(
        modules=_USED_REGION_MODULES,
        header=_USED_REGION_HEADER,
        probes={
            "inner-value": accepted("Geo::Inner(y = 1)", "record tl::Geo::Inner\n  y: int"),
            **type_positions("inner", "Geo::Inner", "tl::Geo::Inner"),
            "point-value": accepted("Geo::Point(x = 1)", "record shapes::Geo::Point\n  x: int"),
            **type_positions("point", "Geo::Point", "shapes::Geo::Point"),
            "neither-value": rejected("Geo::f()", UnknownMemberError, "Geo::f"),
        },
    ),
    "use-region-and-imported-type-beside-a-def-path": Scenario(
        modules=_USED_REGION_MODULES,
        header=(*_USED_REGION_HEADER, "def Geo::f() -> int = 1"),
        probes={"own-function": accepted("Geo::f", "() -> int")},
    ),
    "use-region-beside-imported-receiver-type": Scenario(
        modules={"shapes": _USED_REGION_LIB, "tl": "record Geo\n  x: int\n"},
        header=_USED_REGION_HEADER,
        probes={
            "receiver": accepted(
                "def Geo::f(self) -> Geo = self\nGeo(x = 1).f()", "record tl::Geo\n  x: int"
            )
        },
    ),
    "use-region-beside-two-imported-types": Scenario(
        modules={
            "m": "record X\n  a: int\n",
            "n": "record X\n  b: int\n",
            "lib": "scope X\n  record Y\n    z: int\nend X\n",
        },
        header=("import m::*", "import n::*", "import lib", "use lib::*"),
        probes={
            **type_positions_rejected("bare", "X", AmbiguousQualificationError),
            "member-value": accepted("X::Y(z = 1)", "record lib::X::Y\n  z: int"),
        },
    ),
    "def-path-without-import": Scenario(
        header=("def Geo::f() -> int = 1",),
        probes={
            "value": rejected("Geo::Point(x = 1)", UnknownMemberError, "Geo::Point"),
            "pattern": rejected(
                "case 1 of\n  | Geo::Point(x) => x\n  | _ => 2", UnknownMemberError, "Geo::Point"
            ),
            "is": rejected("1 is Geo::Point", UnknownMemberError, "Geo::Point"),
            **type_positions_rejected("point", "Geo::Point", UnknownMemberError),
        },
    ),
    "def-path-beside-imported-region": Scenario(
        modules={"shapes": _DEFPATH_LIB},
        header=("import shapes::*", "def Geo::f() -> int = 1"),
        probes={
            "pattern": accepted(
                "fn(p: Geo::Point) => case p of\n  | Geo::Point(x) as w => w",
                "shapes::Geo::Point -> shapes::Geo::Point",
            ),
            **type_positions("point", "Geo::Point", "shapes::Geo::Point"),
        },
    ),
    "inner-def-path-beside-length-three-import": Scenario(
        modules={"shapes": _DEFPATH3_LIB},
        header=("import shapes::*", "def Geo::In::f() -> int = 1"),
        probes={
            **type_positions("point", "Geo::In::Point", "shapes::Geo::In::Point"),
            "nope-value": rejected("Geo::In::Nope(x = 1)", UnknownMemberError, "Geo::In::Nope"),
            **type_positions_rejected("nope", "Geo::In::Nope", UnknownMemberError),
            "sibling-is": rejected("1 is Geo::Other", UnknownMemberError, "Geo::Other"),
            **type_positions_rejected("sibling", "Geo::Other", UnknownMemberError),
        },
    ),
    "chain-beside-own-region": Scenario(
        modules={"shapes": _CHAIN_GEO},
        header=("import shapes::*", _CHAIN_LOCAL_DEEP),
        probes=_chain_probes("", "shapes::"),
    ),
    "chain-declared": Scenario(header=(_CHAIN_GEO,), probes=_chain_probes("", "")),
    "chain-anchored-at-the-own-root": Scenario(
        header=(_CHAIN_GEO,), probes=_chain_probes("::", "")
    ),
    "chain-through-a-route": Scenario(
        modules={"lib": _CHAIN_GEO},
        header=("import lib",),
        probes=_chain_probes("lib::", "lib::"),
    ),
    "chain-anchored-at-a-module": Scenario(
        modules={"lib": _CHAIN_GEO},
        header=("import lib",),
        probes=_chain_probes("/lib::", "lib::"),
    ),
    "own-partial-path-beside-a-same-named-route": Scenario(
        modules={"Geo": "let x = 1\n"},
        header=("import Geo", _CHAIN_NESTED_LOCAL),
        probes={
            "value": rejected(
                "Geo::Shape::Missing::X", UnknownMemberError, "Geo::Shape::Missing::X"
            ),
            **type_positions_rejected("type", "Geo::Shape::Missing::X", UnknownMemberError),
        },
    ),
    "own-length-four-path-beside-a-route": Scenario(
        modules={"pkg/Foo": _LEN4_LIB},
        header=("import pkg/Foo", _LEN4_LOCAL_SAME_LEAF),
        probes={
            **type_positions("member", "Foo::E::A::Z", "Foo::E::A::Z"),
        },
    ),
    "route-supplies-a-leaf-the-own-path-lacks": Scenario(
        modules={"pkg/Foo": _LEN4_LIB},
        header=("import pkg/Foo", _LEN4_LOCAL_DIFFERENT_LEAF),
        probes={
            "value": accepted("Foo::E::A::Z(v = 1)", "record pkg/Foo::E::A::Z\n  v: int"),
            **type_positions("member", "Foo::E::A::Z", "pkg/Foo::E::A::Z"),
            "own-value": accepted("Foo::E::A::Q(w = 1)", "record Foo::E::A::Q\n  w: int"),
        },
    ),
}


class TestQualifierPositionMatrix:
    """Each qualifier's decision is the same in every position, mode and REPL grouping."""

    @pytest.mark.parametrize("scenario", scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)
