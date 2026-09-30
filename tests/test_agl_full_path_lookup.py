"""A spelling is looked up as its whole path, against every source.

A scope path is part of a declaration's name, so a scope region never
hides, stops or competes with anything: an own ``scope Agent`` and an
imported ``Agent::f`` combine, and an own ``scope Geo`` leaves an imported
type ``Geo`` visible. Among the declarations one full path reaches, the
module's own wins; otherwise one distinct imported declaration is selected,
and two are ambiguous. Inside ``scope S1::S2`` a spelling ``p`` tries
``S1::S2::p``, then ``S1::p``, then ``p``, each step reading the own
declarations at that path and the contributions anchored at or above it.
``::p`` reads the own root alone, ``/m::p`` that module alone, and a module
route competes like any other contribution. Nothing found is a hidden member
when a ``hiding`` removed the path, an unknown member when its qualifier names
something visible, else an unknown qualifier.

Every probe is checked in file mode and in every legal REPL grouping of its
scenario's header (see :mod:`tests.agl.qualifier_support`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import AglTypeError, HiddenMemberError, ReferencedMemberError
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.scope.symbols import (
    AglScopeError,
    AmbiguousConstructorError,
    AmbiguousQualificationError,
    DeclarationSelection,
    DuplicateDeclarationError,
    OwnerMemberSelection,
    TypeArgumentsError,
    UnknownMemberError,
    UnknownQualifierError,
)
from agm.agl.syntax.types import NameT
from agm.agl.syntax.visitor import walk
from tests.agl.module_graph import resolve_inline_entry
from tests.agl.qualifier_support import (
    Probe,
    Scenario,
    accepted,
    assert_file_resolves_like_inline_entry,
    assert_scenario,
    file_params,
    info,
    info_rejected,
    rejected,
    scenario_params,
    type_positions,
    type_positions_rejected,
)

_M = (
    "scope Agent\n  def f() -> int = 1\n  enum Mood\n    | Up\n    | Down\nend Agent\n"
    "\n"
    "record Geo\n  x: int\nrecord Box[T]\n  v: T\nexception Oops\n  code: int\n"
)
_N = 'scope Agent\n  def f() -> text = "n"\nend Agent\n\nrecord Geo\n  y: text\n'
_OWN_AGENT = "scope Agent\n  def g() -> int = 2\nend Agent"
_OWN_GEO = (
    "scope Geo\n  def make() -> int = 3\nend Geo\n\nscope Box\n  let b = 5\nend Box\n"
    "\n"
    "scope Oops\n  let o = 6\nend Oops"
)
_S = "record R\n  y: text\nenum E\n  | A(y: text)\n  | B\n"
_ROUTED = "record X\n  a: int\nrecord Y\n  b: int\n"
_COLOR = "enum Color\n  | Red\n  | Green\n"
_SHADE = "enum Color\n  | Dark\n  | Light\n"
_TL = "record Geo\n  x: int\nrecord Geo::Inner\n  y: int\n"
_TL_LOCAL = "scope s\n  record Geo\n    x: int\n  record Geo::Inner\n    y: int\nend s"
_COUNTED = (
    "scope Agent\n  var cnt: int = 0\n  let fixed: int = 1\n"
    "  enum Mode\n    | On\n    | Off\nend Agent\n"
)
_OPT = "enum Opt[T]\n  | Som(v: T)\n  | Non\n"
_GENERIC = "record Outer\n  a: int\nenum Outer::Inner[T]\n  | A(v: T)\n  | B\n"
_NESTED_UNDER_GENERICS = (
    "record Box[T]\n  v: T\nrecord Box::Inner\n  x: int\n"
    "enum E[T]\n  | A(a: T)\n  | B\nrecord E::Inner\n  x: int\n"
    "type Al[T] = Box[T]\nrecord Al::Own\n  x: int\n"
    "enum Col\n  | Red\n  | Blue\nenum Two[A, B]\n  | L(l: A)\n  | R(r: B)"
)
_NESTED_Y = "scope G\n  record Y\n    b: int\nend G\n"
_NESTED_Y_CALL = "lib::G::Y(b = 1).m()"
_GEO_REGION = "scope Geo\n  record Point\n    x: int\nend Geo"
_GEO_TYPE = "record Geo\n  y: int\nrecord Geo::Point\n  x: int"
_REFERENCING_ENUM = "enum E\n  | Geo::Point\n  | Other"
_REFERENCING_ENUM_IDENTITY = "enum E\n  | Point(x: int)\n  | Other"
_REFERENCED_MEMBER_PROBES = {
    "injected-value": accepted("let e: E = Point(x = 1)\ne", _REFERENCING_ENUM_IDENTITY),
    "injected-pattern": accepted(
        "let e: E = E::Other\ncase e of\n  | Point(x) => x\n  | Other => 0", "int"
    ),
    "enum-annotation": accepted("fn(e: E) => e is Other", "E -> bool"),
}

_USE_M = (
    "scope Agent\n  def f() -> int = 1\n  def both() -> int = 10\n  def dup() -> int = 11\n"
    "  record P\n    x: int\n  record Q\n    x: int\nend Agent"
)
_USE_N = (
    'scope Agent\n  def n() -> text = "n"\n  def both() -> text = "n"\n'
    '  def dup() -> text = "d"\nend Agent'
)
_USE_O = "scope Agent\n  def dup() -> bool = true\nend Agent"
_USE_OWN = (
    "scope Agent\n  def g() -> int = 2\n  def both() -> bool = true\n"
    "  record Q\n    y: bool\nend Agent"
)
_USE_LIB = "def f() -> int = 1\ndef both() -> int = 10\nrecord P\n  x: int"
_USE_OWN_LIB = "scope lib\n  def g() -> int = 2\n  def both() -> bool = true\nend lib"


_EXPOSED_A = "scope A\n\n  scope X\n    def f() -> int = 1\n    def h() -> int = 1\n  end X\nend A"
_EXPOSED_B = "scope B\n\n  scope X\n    def g() -> int = 2\n    def h() -> int = 2\n  end X\nend B"
_ALIASED_ENUM = "enum E\n  | Red\n  | Blue(v: int)\ntype C = E\n"
_ONE_RED = "record one::E::Red"
_ONE_BLUE = "record one::E::Blue\n  v: int"
_OTHER_RED = "enum F\n  | Red\n"
_OUTSIDE_THE_REGION = "scope r\n  use C::*\n  let v = Red\nend r\n\nRed"
_OWN_ALIASED_ENUM = ("scope S\n  enum E\n    | Red\n    | Blue(v: int)\nend S", "type C = S::E")
_REFERENCED_R = "record R\n  x: int\nenum A\n  | ::R\n  | B\n"
_TWICE_REFERENCED_P = "record P\n  x: int\nenum A\n  | ::P\n  | Q\nenum B\n  | ::P\n  | Z\n"
_USES_X = "use A::*\n  use B::*\n  use X::*"


_ALIASED_BASE = (
    "record Base\n  x: int\nrecord Base::Inner\n  y: int\nrecord Base::Inner::Deep\n  z: int\n"
    "def Base::f() -> int = 1\ndef Base::m(self) -> int = self.x\n"
    "\n"
    "scope Base\n  let b = 5\nend Base\n\nenum Base::Shape\n  | Sq\n  | Ci\n"
    "record Other\n  w: int\nrecord Other::X\n  v: int\ntype Base::Al = Other\n"
    "type Geo = Base\n"
)
_INNER = "record al::Base::Inner\n  y: int"
_DEEP = "record al::Base::Inner::Deep\n  z: int"
_OTHER_X = "record al::Other::X\n  v: int"
_OWN_BASE = "record Base\n  x: int\nrecord Base::Inner\n  y: int\ndef Base::f() -> int = 1"
_GENERIC_BASE = "record Box[T]\n  v: T\nrecord Box::Inner\n  x: int\ntype Al[T] = Box[T]\n"


def _beneath_alias_probes(alias: str) -> dict[str, Probe]:
    """*alias* of ``al``'s ``Base`` reaching every path beneath ``Base``, in every position."""
    return {
        "nested-value": accepted(f"{alias}::Inner(y = 1)", _INNER),
        **type_positions("nested", f"{alias}::Inner", "al::Base::Inner"),
        "nested-pattern": accepted(
            f"case al::Base::Inner(y = 1) of\n  | {alias}::Inner(y) => y", "int"
        ),
        "member-is": accepted(
            f"fn(p: al::Base::Shape) => p is {alias}::Shape::Sq", "al::Base::Shape -> bool"
        ),
        "member-narrow": accepted(
            f"fn(p: al::Base::Shape) => p as? {alias}::Shape::Sq",
            "al::Base::Shape -> std/option::Option[al::Base::Shape::Sq]",
        ),
        "member-pattern": accepted(
            f"fn(p: al::Base::Shape) => case p of\n  | {alias}::Shape::Sq => 1\n  | _ => 2",
            "al::Base::Shape -> int",
        ),
        "deeper-value": accepted(f"{alias}::Inner::Deep(z = 1)", _DEEP),
        **type_positions("deeper", f"{alias}::Inner::Deep", "al::Base::Inner::Deep"),
        "function": accepted(f"{alias}::f", "() -> int"),
        "method": accepted(f"{alias}::m", "al::Base -> int"),
        "binding": accepted(f"{alias}::b", "int"),
        "through-a-nested-alias": accepted(f"{alias}::Al::X(v = 1)", _OTHER_X),
        **type_positions("through-a-nested-alias", f"{alias}::Al::X", "al::Other::X"),
        "unknown-member": rejected(f"{alias}::Nope(y = 1)", UnknownMemberError, f"{alias}::Nope"),
        "unknown-nested-member": rejected(
            f"{alias}::Inner::Nope(y = 1)", UnknownMemberError, f"{alias}::Inner::Nope"
        ),
        "type-arguments-on-a-plain-alias": rejected(
            f"{alias}[int]::Inner(y = 1)", TypeArgumentsError, f"{alias}[int]"
        ),
    }


_RENAMED = _COLOR + "record Pt\n  x: int\nrecord Pt::In\n  y: int\ntype P = Pt\n"
_LOCAL_L = "scope s\n  record L\n    x: int\n  record L::In\n    y: int\nend s\n\n"
_ROOT_L = "record L\n  x: int\nrecord L::In\n  y: int\n\n"


def _renamed_type_probes(
    local_use: str, local: str = _LOCAL_L, local_path: str = "s::L"
) -> dict[str, Probe]:
    """``C``, ``R`` and ``Q`` renaming ``e``'s ``Color``, ``Pt`` and alias ``P``; *local_use*
    renames the own *local_path* that *local* declares to ``M`` inside a region."""
    in_t = local + "scope t\n  {use}\n  let v = {value}\nend t\n\nt::v"
    return {
        **type_positions("enum", "C", "e::Color"),
        "enum-not-a-value": rejected(
            "C", AglTypeError, "C", type_entry="enum e::Color\n  | Red\n  | Green"
        ),
        "enum-member": accepted("C::Red", "record e::Color::Red"),
        "enum-pattern": accepted(
            "fn(p: e::Color) => case p of\n  | C::Red => 1\n  | _ => 2", "e::Color -> int"
        ),
        "enum-is": accepted("fn(p: e::Color) => p is C::Red", "e::Color -> bool"),
        "enum-in-a-region": accepted(
            "scope r\n  use e::Color as K\n  let v: K = K::Red\nend r\n\nr::v",
            "enum e::Color\n  | Red\n  | Green",
        ),
        **type_positions("record", "R", "e::Pt"),
        "record-constructor": accepted("R(x = 1)", "record e::Pt\n  x: int"),
        "record-own-spelling": accepted("R::R(x = 1)", "record e::Pt\n  x: int"),
        "record-own-spelling-pattern": accepted(
            "fn(p: e::Pt) => case p of\n  | R::R(x) => x", "e::Pt -> int"
        ),
        "record-nested": accepted("R::In(y = 1)", "record e::Pt::In\n  y: int"),
        **type_positions("record-nested", "R::In", "e::Pt::In"),
        "record-unknown-member": rejected("R::Nope", UnknownMemberError, "R::Nope"),
        **type_positions("alias", "Q", "e::Pt"),
        "alias-own-spelling": accepted("Q::Q(x = 1)", "record e::Pt\n  x: int"),
        "alias-own-spelling-pattern": accepted(
            "fn(p: e::Pt) => case p of\n  | Q::Q(x) => x", "e::Pt -> int"
        ),
        "alias-nested": accepted("Q::In(y = 1)", "record e::Pt::In\n  y: int"),
        "local-constructor": accepted(
            in_t.format(use=local_use, value="M(x = 1)"), f"record {local_path}\n  x: int"
        ),
        "local-nested": accepted(
            in_t.format(use=local_use, value="M::In(y = 1)"),
            f"record {local_path}::In\n  y: int",
        ),
        "local-annotation": accepted(
            in_t.format(use=local_use, value="fn(p: M) => p"), f"{local_path} -> {local_path}"
        ),
        "info-enum": info(
            "C", "C is an enum type.\nType:\n  enum e::Color\n    | Red\n    | Green"
        ),
        "info-record": info("R", "R is a constructor.\nSignature:\n  R(x: int) -> e::Pt"),
        "info-alias": info("Q", "Q is a type alias.\nType:\n  type Q = Pt"),
    }


_OWN_ROOT_X = "scope X\n  def f() -> int = 1\n  def h() -> int = 1\nend X"
_OWN_NESTED_X = (
    'scope A\n\n  scope X\n    def k() -> int = 3\n    def h() -> text = "a"\n  end X\nend A'
)


def _in_q(uses: str, value: str) -> str:
    """A region ``Q`` with *uses* and its own ``X``; its ``let v`` reads *value*, then ``Q::v``."""
    inner = "  scope X\n    def g() -> int = 2\n    def h() -> bool = true\n  end X"
    return f"scope Q\n  {uses}\n{inner}\n  let v = {value}\nend Q\n\nQ::v"


def _in_region(use: str, value: str) -> str:
    """A region whose ``use`` exposes names that its ``let v`` reads, then ``r::v``."""
    return f"scope r\n  {use}\n  let v = {value}\nend r\n\nr::v"


def _in_nested(use: str, value: str) -> str:
    """A region whose ``use`` exposes names that a nested region's ``let v`` reads."""
    return f"scope r\n  {use}\n\n  scope q\n    let v = {value}\n  end q\nend r\n\nr::q::v"


_LATER_M = "scope Agent\n  def f() -> int = 1\n  var cnt: int = 0\nend Agent\n"
_LATER_OWN_SCOPE = 'scope Agent\n  def h() -> text = "own"\nend Agent'
_LATER_OWN_BINDINGS = 'scope Agent\n  let cnt: text = "own"\n  def f() -> text = "own"\nend Agent'
_X_A = "scope X\n  def a() -> int = 1\nend X\n"
_X_B = 'scope X\n  def b() -> text = "2"\nend X\n'
_ALIASED_MODULE = f"{_USE_LIB}\nscope Deep\n  def d() -> int = 4\nend Deep\n"
_WRAPPED_COLOR = (
    "enum Color\n  | Red\n  | Blue\nenum Wrap\n  | wrap(shade: Color)\n"
    "def rank(c: Color) -> int = 1\n"
)
_SIGNAL = "enum Signal\n  | Red\n  | Green\n"
_BOUND_GEO = "record R\n  x: int\nlet Geo: R = R(x = 1)\n"
_SCOPED_COLOR = "scope S\n  enum Color\n    | Red\n    | Blue\nend S"


def _branch(value: str, *, region: bool) -> str:
    """A ``wrap(Red)`` branch reading *value*, at the root or in region ``S`` using ``Color``."""
    pad = "  " if region else ""
    body = (
        f"{pad}def a(w: Wrap) -> int =\n{pad}  case w of\n{pad}    | wrap(Red) => {value}\n"
        f"{pad}    | wrap(Blue) => 0\n"
    )
    if not region:
        return f"{body}a(wrap(Color::Blue))"
    return f"scope S\n  use Color::*\n{body}end S\n\nS::a(wrap(Color::Blue))"


_SCENARIOS = {
    "a-use-combines-an-own-scope-with-an-imported-one": Scenario(
        modules={"M": _USE_M},
        header=("import M::*", _USE_OWN),
        probes={
            "imported-member": accepted(_in_region("use Agent::*", "f()"), "int"),
            "own-member": accepted(_in_region("use Agent::*", "g()"), "int"),
            "own-member-wins-its-path": accepted(_in_region("use Agent::*", "both()"), "bool"),
            "imported-type-value": accepted(
                _in_region("use Agent::*", "P(x = 1)"), "record M::Agent::P\n  x: int"
            ),
            "imported-type-annotation": accepted(
                _in_region("use Agent::*", "fn(p: P) => p.x"), "M::Agent::P -> int"
            ),
            "own-type-wins-its-path-in-annotation": accepted(
                _in_region("use Agent::*", "fn(q: Q) => q.y"), "Agent::Q -> bool"
            ),
            "own-type-wins-its-path-as-constructor": accepted(
                _in_region("use Agent::*", "Q(y = true)"), "record Agent::Q\n  y: bool"
            ),
            "own-type-wins-its-path-in-pattern": accepted(
                _in_region("use Agent::*", "case Q(y = true) of\n    | Q(y) => y"), "bool"
            ),
            "selected-from-both": accepted(_in_region("use Agent::{f, g}", "f() + g()"), "int"),
            "hiding-an-imported-member": rejected(
                _in_region("use Agent::* hiding f", "f()"), AglScopeError, "f"
            ),
            "selecting-a-member-neither-declares": rejected(
                _in_region("use Agent::{h}", "1"), UnknownMemberError, "use Agent::{h}"
            ),
        },
    ),
    "a-use-combines-an-own-scope-with-two-imported-ones": Scenario(
        modules={"M": _USE_M, "N": _USE_N},
        header=("import M::*", "import N::*", _USE_OWN),
        probes={
            "one-import-and-own": accepted(_in_region("use Agent::*", "f() + g()"), "int"),
            "the-other-import": accepted(_in_region("use Agent::*", "n()"), "text"),
            "own-member-wins-its-path": accepted(_in_region("use Agent::*", "both()"), "bool"),
            "two-imported-members-clash-where-used": rejected(
                _in_region("use Agent::*", "dup()"), AmbiguousQualificationError, "dup"
            ),
        },
    ),
    "a-use-combines-imported-scopes": Scenario(
        modules={"M": _USE_M, "N": _USE_N, "O": _USE_O},
        header=("import M::*", "import N::*", "import O::*"),
        probes={
            "one-import": accepted(_in_region("use Agent::*", "f()"), "int"),
            "the-other-import": accepted(_in_region("use Agent::*", "n()"), "text"),
            "two-imported-members-clash-where-used": rejected(
                _in_region("use Agent::*", "both()"), AmbiguousQualificationError, "both"
            ),
            "three-imported-members-clash-where-used": rejected(
                _in_region("use Agent::*", "dup()"), AmbiguousQualificationError, "dup"
            ),
        },
    ),
    "a-use-combines-an-own-scope-with-a-module-route": Scenario(
        modules={"lib": _USE_LIB},
        header=("import lib", _USE_OWN_LIB),
        probes={
            "routed-member": accepted(_in_region("use lib::*", "f()"), "int"),
            "own-member": accepted(_in_region("use lib::*", "g()"), "int"),
            "own-member-wins-its-path": accepted(_in_region("use lib::*", "both()"), "bool"),
            "routed-type-value": accepted(
                _in_region("use lib::*", "P(x = 1)"), "record lib::P\n  x: int"
            ),
        },
    ),
    "a-use-of-an-imported-alias-reaches-its-target-members": Scenario(
        modules={"one": _ALIASED_ENUM},
        header=("import one::{C}",),
        probes={
            "member": accepted(_in_region("use C::*", "Red"), _ONE_RED),
            "member-with-fields": accepted(_in_region("use C::*", "Blue(v = 1)"), _ONE_BLUE),
            "member-pattern": accepted(
                _in_region(
                    "use C::*", "case Blue(v = 2) as C of\n    | Red => 1\n    | Blue(v) => v"
                ),
                "int",
            ),
            "selected-member": accepted(_in_region("use C::{Red}", "Red"), _ONE_RED),
            "routed-alias": accepted(_in_region("use one::C::*", "Red"), _ONE_RED),
            "hidden-member": rejected(
                _in_region("use C::* hiding Red", "Red"), AglScopeError, "Red"
            ),
            "outside-the-region": rejected(_OUTSIDE_THE_REGION, AglScopeError, "Red"),
        },
    ),
    "a-use-of-an-alias-chain-reaches-its-last-target": Scenario(
        modules={"one": f"{_ALIASED_ENUM}type D = C\ntype N = int\n"},
        header=("import one::{D, N}",),
        probes={
            "member": accepted(_in_region("use D::*", "Red"), _ONE_RED),
            "alias-of-a-builtin-type": accepted(_in_region("use N::*", "1"), "int"),
        },
    ),
    "an-alias-member-clashes-with-an-imported-one": Scenario(
        modules={"one": _ALIASED_ENUM, "two": _OTHER_RED},
        header=("import one::{C}", "import two::*", "use C::*"),
        probes={
            "same-name": rejected("[Red]", AmbiguousConstructorError, "Red"),
            "other-member": accepted("Blue(v = 1)", _ONE_BLUE),
        },
    ),
    "a-region-alias-use-is-read-before-the-root": Scenario(
        modules={"one": _ALIASED_ENUM, "two": _OTHER_RED},
        header=("import one::{C}", "import two::*"),
        probes={
            "alias-member": accepted(_in_region("use C::*", "Red"), _ONE_RED),
            "enum-member": accepted(_in_region("use one::E::*", "Red"), _ONE_RED),
        },
    ),
    "a-use-of-an-own-alias-reaches-its-target-members": Scenario(
        header=_OWN_ALIASED_ENUM,
        probes={
            "member": accepted(_in_region("use C::*", "Red"), "record S::E::Red"),
            "module-root-alias": accepted(_in_region("use ::C::*", "Red"), "record S::E::Red"),
            "unselected-member": rejected(
                _in_region("use C::{Red}", "Blue(v = 1)"), AglScopeError, "Blue"
            ),
            "outside-the-region": rejected(_OUTSIDE_THE_REGION, AglScopeError, "Red"),
        },
    ),
    "an-own-alias-wins-its-path-over-an-imported-one": Scenario(
        modules={"one": _ALIASED_ENUM},
        header=("import one::{C}", *_OWN_ALIASED_ENUM),
        probes={"member": accepted(_in_region("use C::*", "Red"), "record S::E::Red")},
    ),
    "an-alias-member-path-is-declared-once": Scenario(
        header=_OWN_ALIASED_ENUM,
        probes={
            "scope-function": rejected(
                "scope C\n  def Red() -> int = 1\nend C",
                DuplicateDeclarationError,
                "def Red() -> int = 1",
            ),
            "scope-binding": rejected(
                "scope C\n  let Blue = 1\nend C", DuplicateDeclarationError, "let Blue = 1"
            ),
            "qualified-function": rejected(
                "def C::Blue() -> int = 1", DuplicateDeclarationError, "def C::Blue() -> int = 1"
            ),
            "qualified-type": rejected("record C::Red", DuplicateDeclarationError, "record C::Red"),
            "another-name": accepted(
                "scope C\n  def Green() -> int = 1\nend C\n\nC::Green()", "int"
            ),
        },
    ),
    "an-alias-declared-after-its-member-path": Scenario(
        header=(_OWN_ALIASED_ENUM[0],),
        probes={
            "scope-function": rejected(
                "scope C\n  def Red() -> int = 1\nend C\n\ntype C = S::E",
                DuplicateDeclarationError,
                "type C = S::E",
            ),
        },
    ),
    "an-alias-of-an-imported-enum-declares-beside-its-member-paths": Scenario(
        modules={"one": _ALIASED_ENUM},
        header=("import one::{E}", "type C = E"),
        probes={
            f"{declared}-{form}-then-{used}": accepted(f"{declaration}\n\n{used}::{name}()", "text")
            for declared in ("C", "E")
            for used in ("C", "E")
            for form, name, declaration in (
                ("region", "Red", f'scope {declared}\n  def Red() -> text = "own"\nend {declared}'),
                ("qualified", "Blue", f'def {declared}::Blue() -> text = "own"'),
            )
        }
        | {
            f"{declared}-type-then-{used}": accepted(
                f"record {declared}::Red\n  q: bool\n{used}::Red(q = true)",
                f"record {declared}::Red\n  q: bool",
            )
            for declared in ("C", "E")
            for used in ("C", "E")
        },
    ),
    "an-alias-of-an-imported-enum-declared-after-its-member-path": Scenario(
        modules={"one": _ALIASED_ENUM},
        header=("import one::{E}",),
        probes={
            f"then-{used}": accepted(
                f'scope C\n  def Red() -> text = "own"\nend C\n\ntype C = E\n\n{used}::Red()',
                "text",
            )
            for used in ("C", "E")
        },
    ),
    "hiding-an-enum-member-path-removes-its-bare-spelling": Scenario(
        modules={"a": _REFERENCED_R},
        header=("import std/prelude::* hiding Option::Some", "import a::* hiding R"),
        probes={
            "prelude-member": rejected("Some(1)", AglScopeError, "Some"),
            "prelude-sibling": accepted("[None, Option::None]", "array[std/option::Option::None]"),
            "referenced-member": rejected("R(x = 1)", AglScopeError, "R"),
            "referencing-sibling": accepted("B", "record a::A::B"),
        },
    ),
    "a-hidden-enum-member-spelling-is-free": Scenario(
        modules={"lib": "def Some() -> int = 1\ndef R() -> int = 2\n", "a": _REFERENCED_R},
        header=(
            "import std/prelude::* hiding Option::Some",
            "import a::* hiding R",
            "import lib::*",
        ),
        probes={
            "prelude-member": accepted("Some()", "int"),
            "referenced-member": accepted("R()", "int"),
        },
    ),
    "another-import-keeps-a-hidden-enum-member": Scenario(
        modules={"a": _REFERENCED_R},
        header=(
            "import std/prelude::* hiding Option::Some",
            "import std/option::{Option}",
            "import a::* hiding R",
            "import a::{R}",
        ),
        probes={
            "prelude-member": accepted(
                "Some(1)", "record std/option::Option::Some[int]\n  value: int"
            ),
            "referenced-member": accepted("R(x = 1)", "record a::R\n  x: int"),
        },
    ),
    "a-use-reads-own-scopes-by-whole-path": Scenario(
        header=(_OWN_ROOT_X, _OWN_NESTED_X),
        probes={
            "outer-and-inner-scope": accepted(_in_q("use X::*", "f() + g()"), "int"),
            "inner-scope-wins-its-path": accepted(_in_q("use X::*", "h()"), "bool"),
            "tail-over-both": accepted(_in_q("use X::{f, g}", "f() + g()"), "int"),
            "alias-over-both": accepted(_in_q("use X as Y", "Y::f() + Y::g()"), "int"),
            "exposed-and-inner-scope": accepted(
                _in_q("use A::*\n  use X::*", "k() + g() + f()"), "int"
            ),
            "inner-scope-wins-over-an-exposed-one": accepted(
                _in_q("use A::*\n  use X::*", "h()"), "bool"
            ),
            "exposed-scope-wins-over-an-outer-one": accepted(
                "scope Q\n  use A::*\n  use X::*\n  let v = h()\nend Q\n\nQ::v", "text"
            ),
        },
    ),
    "a-record-two-imported-enums-reference-is-one-declaration": Scenario(
        modules={"lib": _TWICE_REFERENCED_P},
        header=("import lib::{A, B}",),
        probes={"bare": accepted("P(x = 1)", "record lib::P\n  x: int")},
    ),
    "a-use-of-an-enum-exposes-only-its-inline-members": Scenario(
        modules={"lib": _TWICE_REFERENCED_P},
        header=("import lib", "use lib::A::*", "use lib::B::*"),
        probes={
            "referenced": rejected("P(x = 1)", AglScopeError, "P"),
            "inline": accepted("Z", "record lib::B::Z"),
        },
    ),
    "an-enum-member-is-no-bare-type-own-or-imported": Scenario(
        modules={"two": "enum Color\n  | Red\n  | Green\n"},
        header=("import two::*", "enum Shade\n  | Dark\n  | Light"),
        probes={
            "own": rejected("fn(x: Dark) => x", AglTypeError, "Dark"),
            "imported": rejected("fn(x: Red) => x", AglTypeError, "Red"),
            "in-a-region": rejected(
                _in_region("use two::*", "fn(x: Red) => x"), AglTypeError, "Red"
            ),
            "unknown": rejected("fn(x: Nowhere) => x", AglTypeError, "Nowhere"),
            "imported-qualified": rejected("fn(x: two::Red) => x", AglTypeError, "two::Red"),
            "own-qualified": rejected("fn(x: ::Dark) => x", AglTypeError, "::Dark"),
        },
    ),
    "a-use-combines-the-scopes-earlier-uses-expose": Scenario(
        header=(_EXPOSED_A, _EXPOSED_B),
        probes={
            "one-exposure-each": accepted(_in_region(_USES_X, "f() + g()"), "int"),
            "two-exposed-members-clash-where-used": rejected(
                _in_region(_USES_X, "h()"), AmbiguousQualificationError, "h"
            ),
            "selected-from-both": accepted(
                _in_region("use A::*\n  use B::*\n  use X::{f, g}", "f() + g()"), "int"
            ),
            "selecting-a-member-neither-declares": rejected(
                _in_region("use A::*\n  use B::*\n  use X::{k}", "1"),
                UnknownMemberError,
                "use X::{k}",
            ),
            "hiding-a-member-both-declare": rejected(
                _in_region("use A::*\n  use B::*\n  use X::* hiding h", "h()"), AglScopeError, "h"
            ),
            "exposed-by-enclosing-and-own-region": accepted(
                "scope r\n  use A::*\n\n  scope q\n    use B::*\n    use X::*\n"
                "    let v = f() + g()\n  end q\nend r\n\nr::q::v",
                "int",
            ),
        },
    ),
    "own-scope-and-imported-scope-combine": Scenario(
        modules={"M": _M},
        header=("import M::*", _OWN_AGENT, _OWN_GEO),
        probes={
            "combined-values": accepted("Agent::f() + Agent::g()", "int"),
            "motivating-example": accepted(
                "def h() -> int = Agent::f() + Agent::g()\n"
                "def k(p: Geo) -> int = Geo::make()\n"
                "h() + k(Geo(x = 1)) + Box::b + Oops::o",
                "int",
            ),
            "imported-applied-type": accepted("fn(p: Box[int]) => p.v", "M::Box[int] -> int"),
            "imported-exception-catch": accepted("try\n  1\ncatch Oops as e =>\n  e.code", "int"),
            "imported-exception-extends": accepted(
                "exception Local extends Oops\nfn(p: Local) => p.code", "Local -> int"
            ),
            "imported-type-repl-entry": rejected(
                "Agent::Mood",
                AglTypeError,
                "Agent::Mood",
                type_entry="enum M::Agent::Mood\n  | Up\n  | Down",
            ),
            "own-member-of-imported-type-path": accepted("Geo::make()", "int"),
            "info-imported-generic-type": info(
                "Box", "Box is a generic record type.\nType:\n  record Box[T]\n    v: T"
            ),
            "info-imported-scoped-enum": info(
                "Agent::Mood",
                "Agent::Mood is an enum type.\nType:\n  enum M::Agent::Mood\n    | Up\n    | Down",
            ),
            **type_positions("imported-geo", "Geo", "M::Geo"),
            **type_positions("imported-scoped", "Agent::Mood", "M::Agent::Mood"),
            "imported-type-cast": accepted("fn(p: Geo) => p as Geo", "M::Geo -> M::Geo"),
            "imported-type-value": accepted("Geo(x = 1)", "record M::Geo\n  x: int"),
            "imported-type-pattern": accepted("case Geo(x = 1) of\n  | Geo(x) => x", "int"),
            "imported-scope-is": accepted(
                "fn(p: Agent::Mood) => p is Agent::Mood::Up", "M::Agent::Mood -> bool"
            ),
            "imported-type-receiver": accepted(
                "def Geo::norm(self) -> int = self.x\nGeo(x = 2).norm()", "int"
            ),
            "missing-member-of-combined-scope": rejected(
                "Agent::h()", UnknownMemberError, "Agent::h"
            ),
            "imported-scope-from-inside-own-region": accepted(
                "scope Agent\n  let v = f() + g()\nend Agent\n\nAgent::v", "int"
            ),
        },
    ),
    "own-declaration-beats-imported-one": Scenario(
        modules={"M": _M},
        header=(
            "import M::*",
            'scope Agent\n  def f() -> text = "own"\nend Agent',
            "record Geo\n  z: text",
        ),
        probes={
            "own-scoped-value": accepted("Agent::f()", "text"),
            "own-type-annotation": accepted("fn(p: Geo) => p.z", "Geo -> text"),
            "own-type-value": accepted('Geo(z = "a")', "record Geo\n  z: text"),
            "own-type-pattern": accepted('case Geo(z = "a") of\n  | Geo(z) => z', "text"),
            "own-type-receiver": accepted(
                'def Geo::m(self) -> text = self.z\nGeo(z = "a").m()', "text"
            ),
            "imported-path-still-qualified": accepted("M::Agent::f()", "int"),
        },
    ),
    "two-imports-clash": Scenario(
        modules={"M": _M, "N": _N},
        header=("import M::*", "import N::*"),
        probes={
            "scoped-value": rejected("Agent::f()", AmbiguousQualificationError, "Agent::f"),
            "type-annotation": rejected("fn(p: Geo) => 1", AmbiguousQualificationError, "Geo"),
            "constructor": rejected("Geo(x = 1)", AmbiguousConstructorError, "Geo"),
            "routed-value": accepted("M::Agent::f()", "int"),
            "info-scoped-value": info_rejected("Agent::f", AmbiguousQualificationError, "Agent::f"),
            "info-type": info_rejected("Geo", AmbiguousConstructorError, "Geo"),
            "info-routed-value": info(
                "M::Agent::f", "M::Agent::f is a function.\nSignature:\n  def M::Agent::f() -> int"
            ),
            "info-unique-scoped-enum": info(
                "Agent::Mood",
                "Agent::Mood is an enum type.\nType:\n  enum M::Agent::Mood\n    | Up\n    | Down",
            ),
        },
    ),
    "two-uses-clash": Scenario(
        modules={"M": _M, "N": _N},
        header=("import M", "import N", "use M::*", "use N::*"),
        probes={
            "scoped-value": rejected("Agent::f()", AmbiguousQualificationError, "Agent::f"),
            "type-annotation": rejected("fn(p: Geo) => p", AmbiguousQualificationError, "Geo"),
            "info-scoped-value": info_rejected("Agent::f", AmbiguousQualificationError, "Agent::f"),
            "info-type": info_rejected("Geo", AmbiguousConstructorError, "Geo"),
            "info-unique-type": info(
                "Box", "Box is a generic record type.\nType:\n  record Box[T]\n    v: T"
            ),
        },
    ),
    "scoped-use-beats-root-own-declaration": Scenario(
        modules={"s": _S},
        header=("import s", "record R\n  x: int"),
        probes={
            "value": accepted(
                'scope r\n  use s::*\n  let v = R(y = "a")\nend r\n\nr::v', "record s::R\n  y: text"
            ),
            "annotation": accepted(
                "scope r\n  use s::*\n  let v = fn(p: R) => p.y\nend r\n\nr::v", "s::R -> text"
            ),
            "pattern": accepted(
                'scope r\n  use s::*\n  let v = case R(y = "a") of\n    | R(y) => y\nend r\n\nr::v',
                "text",
            ),
            "receiver": accepted(
                "scope r\n  use s::*\n  def R::m(self) -> text = self.y\n"
                '  let v = R(y = "a").m()\nend r\n\nr::v',
                "text",
            ),
            "root-own-outside-region": accepted("R(x = 1)", "record R\n  x: int"),
            "own-in-region-beats-use": accepted(
                "scope r\n  use s::*\n  record R\n    w: int\n  let v = R(w = 1)\nend r\n\nr::v",
                "record r::R\n  w: int",
            ),
        },
    ),
    "route-and-own-path": Scenario(
        modules={"lib": _ROUTED},
        header=(
            "import lib",
            "scope lib\n  record X\n    c: text\nend lib",
            "record Z\n  z: int",
        ),
        probes={
            "own-path-beats-route": accepted('lib::X(c = "a")', "record lib::X\n  c: text"),
            "route-supplies-missing-member": accepted("lib::Y(b = 1)", "record lib::Y\n  b: int"),
            "anchored-route-reads-module-only": accepted(
                "/lib::X(a = 1)", "record lib::X\n  a: int"
            ),
            "unknown-member-of-both": rejected("lib::W", UnknownMemberError, "lib::W"),
            "own-root": accepted("::Z(z = 1)", "record Z\n  z: int"),
            "own-root-scope": accepted('::lib::X(c = "a")', "record lib::X\n  c: text"),
            "own-root-scope-ignores-route": rejected(
                "::lib::Y(b = 1)", UnknownMemberError, "::lib::Y"
            ),
            "unknown-qualifier": rejected("nosuch::W", UnknownQualifierError, "nosuch::W"),
            "route-takes-no-type-arguments": rejected(
                "lib[int]::Y(b = 1)", TypeArgumentsError, "lib[int]"
            ),
            **type_positions("route-type", "lib::Y", "lib::Y"),
            **type_positions("own-path-type", "lib::X", "lib::X"),
            **type_positions("own-root-type", "::lib::X", "lib::X"),
            **type_positions_rejected("own-root-ignores-route", "::lib::Y", UnknownMemberError),
        },
    ),
    "an-alias-segment-stands-for-every-path-beneath-its-target": Scenario(
        modules={"al": _ALIASED_BASE},
        header=("import al::*",),
        probes={
            **_beneath_alias_probes("Geo"),
            "literal-nested-alias": accepted("Base::Al::X(v = 1)", _OTHER_X),
            "routed": accepted("al::Geo::Inner(y = 1)", _INNER),
            "routed-deeper": accepted("al::Geo::Inner::Deep(z = 1)", _DEEP),
            "own-spelling": accepted("Geo::Geo(x = 1)", "record al::Base\n  x: int"),
            "info-nested": info(
                "Geo::Inner",
                "Geo::Inner is a constructor.\nSignature:\n  Geo::Inner(y: int) -> al::Base::Inner",
            ),
            "info-function": info(
                "Geo::f", "Geo::f is a function.\nSignature:\n  def Geo::f() -> int"
            ),
            "info-binding": info(
                "Geo::b", "Geo::b is a binding.\nBinding:\n  let Geo::b\nType:\n  int\nValue:\n  5"
            ),
            "info-nested-enum": info(
                "Geo::Shape",
                "Geo::Shape is an enum type.\nType:\n  enum al::Base::Shape\n    | Sq\n    | Ci",
            ),
        },
    ),
    "an-alias-chain-stands-for-the-paths-beneath-its-last-target": Scenario(
        modules={"al": _ALIASED_BASE},
        header=("import al::*", "type H = Geo"),
        probes=_beneath_alias_probes("H"),
    ),
    "an-own-alias-segment-stands-for-an-imported-target-path": Scenario(
        modules={"al": _ALIASED_BASE},
        header=("import al", "type G = al::Base"),
        probes=_beneath_alias_probes("G"),
    ),
    "an-own-alias-segment-stands-for-an-own-target-path": Scenario(
        header=(_OWN_BASE, "type G = Base"),
        probes={
            "nested-value": accepted("G::Inner(y = 1)", "record Base::Inner\n  y: int"),
            **type_positions("nested", "G::Inner", "Base::Inner"),
            "function": accepted("G::f", "() -> int"),
            "own-path-beside-the-target-paths": accepted(
                "record G::Own\n  o: int\nG::Own(o = 1)", "record G::Own\n  o: int"
            ),
        },
    ),
    "a-generic-alias-segment-takes-type-arguments-only-for-an-inline-member": Scenario(
        modules={"lib": _GENERIC_BASE},
        header=("import lib::*",),
        probes={
            "unapplied-value": accepted("Al::Inner(x = 1)", "record lib::Box::Inner\n  x: int"),
            **type_positions("unapplied", "Al::Inner", "lib::Box::Inner"),
            "applied-value": rejected("Al[int]::Inner(x = 1)", TypeArgumentsError, "Al[int]"),
            **type_positions_rejected("applied", "Al[int]::Inner", TypeArgumentsError, "Al[int]"),
        },
    ),
    "hiding-a-target-path-hides-it-through-an-alias-declared-beside-the-import": Scenario(
        modules={"al": _ALIASED_BASE},
        header=("import al::* hiding Base::Inner, Base::f", "type G = Base"),
        probes={
            "hidden-value": rejected("G::Inner(y = 1)", HiddenMemberError, "G::Inner"),
            **type_positions_rejected("hidden", "G::Inner", HiddenMemberError),
            "hidden-deeper": rejected("G::Inner::Deep(z = 1)", HiddenMemberError, "G::Inner::Deep"),
            "hidden-function": rejected("G::f", HiddenMemberError, "G::f"),
            "unhidden-binding": accepted("G::b", "int"),
            "unhidden-method": accepted("G::m", "al::Base -> int"),
        },
    ),
    "an-alias-declares-every-path-beneath-its-target-once": Scenario(
        header=(_OWN_BASE, "type G = Base"),
        probes={
            "nested-type": rejected(
                "record G::Inner\n  q: int", DuplicateDeclarationError, "record G::Inner\n  q: int"
            ),
            "function": rejected(
                "def G::f() -> int = 2", DuplicateDeclarationError, "def G::f() -> int = 2"
            ),
            "scope-function": rejected(
                "scope G\n  def f() -> int = 2\nend G",
                DuplicateDeclarationError,
                "def f() -> int = 2",
            ),
            "another-name": accepted("def G::g() -> int = 2\nG::g", "() -> int"),
        },
    ),
    "hiding-a-deeper-target-path-hides-only-it-through-an-alias": Scenario(
        modules={"al": _ALIASED_BASE},
        header=("import al::* hiding Base::Inner::Deep", "type G = Base"),
        probes={
            "nested": accepted("G::Inner(y = 1)", _INNER),
            "deeper-value": rejected("G::Inner::Deep(z = 1)", HiddenMemberError, "G::Inner::Deep"),
            **type_positions_rejected("deeper", "G::Inner::Deep", HiddenMemberError),
        },
    ),
    "an-alias-cycle-stands-for-no-path": Scenario(
        header=(),
        probes={
            "value": rejected("type C = D\ntype D = C\nC::y", UnknownMemberError, "C::y"),
            "annotation": rejected(
                "type C = D\ntype D = C\nfn(p: C::y) => 1", UnknownMemberError, "C::y"
            ),
        },
    ),
    "a-hidden-target-path-leaves-its-alias-path-free": Scenario(
        modules={"al": _ALIASED_BASE},
        header=("import al::* hiding Base::Inner", "type G = Base"),
        probes={
            "own-declaration": accepted(
                "record G::Inner\n  q: int\nG::Inner(q = 1)", "record G::Inner\n  q: int"
            ),
        },
    ),
    "a-braced-use-renames-a-type": Scenario(
        modules={"e": _RENAMED},
        header=("import e", "use e::{Color as C, Pt as R, P as Q}"),
        probes=_renamed_type_probes("use s::{L as M}"),
    ),
    "an-unbraced-use-renames-a-type-like-a-braced-one": Scenario(
        modules={"e": _RENAMED},
        # One entry: a REPL entry using a module supersedes its earlier uses.
        header=("import e", "use e::Color as C\nuse e::Pt as R\nuse e::P as Q"),
        probes=_renamed_type_probes("use s::L as M"),
    ),
    "a-one-segment-use-renames-a-type-like-a-braced-one": Scenario(
        modules={"e": _RENAMED},
        header=("import e::*", "use Color as C\nuse Pt as R\nuse P as Q"),
        probes=_renamed_type_probes("use L as M", _ROOT_L, "L"),
    ),
    "alias-segments": Scenario(
        modules={"e": _COLOR, "f": _SHADE},
        header=("import e", "import f", "use e::Color as C"),
        probes={
            "use-alias-member": accepted("C::Red", "record e::Color::Red"),
            **type_positions("use-alias", "C::Green", "e::Color::Green"),
            **type_positions_rejected("unknown-alias-member", "C::Blue", UnknownMemberError),
            "use-alias-pattern": accepted(
                "fn(p: e::Color) => case p of\n  | C::Red => 1\n  | _ => 2", "e::Color -> int"
            ),
            "type-alias-member": accepted("type D = f::Color\nD::Dark", "record f::Color::Dark"),
            "nearer-use-alias": accepted(
                "scope r\n  use f::Color as C\n  let v = C::Dark\nend r\n\nr::v",
                "record f::Color::Dark",
            ),
            "farther-use-alias-still-found": accepted(
                "scope r\n  use f::Color as C\n  let v = C::Red\nend r\n\nr::v",
                "record e::Color::Red",
            ),
            "unknown-alias-member": rejected("C::Blue", UnknownMemberError, "C::Blue"),
            "info-routed-enum": info(
                "e::Color",
                "e::Color is an enum type.\nType:\n  enum e::Color\n    | Red\n    | Green",
            ),
            "info-use-alias-member": info(
                "C::Red", "C::Red is a constructor.\nSignature:\n  C::Red() -> e::Color::Red"
            ),
            "info-unknown-alias-member": info_rejected("C::Blue", UnknownMemberError, "C::Blue"),
        },
    ),
    "generic-use-alias-segments": Scenario(
        modules={"lib": _OPT},
        header=("import lib", "use lib::Opt as B"),
        probes={
            "applied-alias-value": accepted(
                "B[int]::Som(v = 1)", "record lib::Opt::Som[int]\n  v: int"
            ),
            **type_positions("applied-alias", "B[int]::Som", "lib::Opt::Som[int]"),
            "applied-scoped-alias": accepted(
                'scope q\n  use lib::Opt as K\n  let v = K[text]::Som(v = "x")\nend q\n\nq::v',
                "record lib::Opt::Som[text]\n  v: text",
            ),
            "wrong-arity-alias": rejected("B[int, text]::Non", TypeArgumentsError, "B[int, text]"),
        },
    ),
    "hidden-unknown-member-and-unknown-qualifier": Scenario(
        modules={"tl": _TL},
        # One entry: a REPL entry re-importing a module supersedes its earlier import.
        header=("import tl\nimport tl::* hiding Geo::Inner",),
        probes={
            "hidden-value": rejected("Geo::Inner(y = 1)", HiddenMemberError, "Geo::Inner"),
            **type_positions_rejected("hidden", "Geo::Inner", HiddenMemberError),
            **type_positions("route-not-hidden", "tl::Geo::Inner", "tl::Geo::Inner"),
            **type_positions_rejected("unknown-member", "Geo::Nope", UnknownMemberError),
            **type_positions_rejected("unknown-qualifier", "Nope::Inner", UnknownQualifierError),
            "route-not-hidden": accepted(
                "tl::Geo::Inner(y = 1)", "record tl::Geo::Inner\n  y: int"
            ),
            "unknown-member": rejected("Geo::Nope(y = 1)", UnknownMemberError, "Geo::Nope"),
            "unknown-qualifier": rejected(
                "Nope::Inner(y = 1)", UnknownQualifierError, "Nope::Inner"
            ),
            "info-hidden": info_rejected("Geo::Inner", HiddenMemberError, "Geo::Inner"),
            "info-route-not-hidden": info(
                "tl::Geo::Inner",
                "tl::Geo::Inner is a constructor.\nSignature:\n"
                "  tl::Geo::Inner(y: int) -> tl::Geo::Inner",
            ),
            "info-unknown-member": info_rejected("Geo::Nope", UnknownMemberError, "Geo::Nope"),
            "info-unknown-qualifier": info_rejected(
                "Nope::Inner", UnknownQualifierError, "Nope::Inner"
            ),
        },
    ),
    "hiding-an-owner-hides-its-members": Scenario(
        modules={"tl": _TL},
        header=("import tl::* hiding Geo",),
        probes={
            "hidden-owner-value": rejected("Geo::Inner(y = 1)", HiddenMemberError, "Geo::Inner"),
            "hidden-owner-annotation": rejected(
                "fn(p: Geo::Inner) => 1", HiddenMemberError, "Geo::Inner"
            ),
        },
    ),
    "use-hiding-an-owner-hides-its-members": Scenario(
        modules={"tl": _TL},
        header=("import tl", "use tl::* hiding Geo"),
        probes={
            "hidden-owner-value": rejected("Geo::Inner(y = 1)", HiddenMemberError, "Geo::Inner"),
        },
    ),
    "scoped-local-use-hiding-an-owner-hides-its-members": Scenario(
        header=(_TL_LOCAL,),
        probes={
            "hidden-owner-value": rejected(
                "scope r\n  use s::* hiding Geo\n\n  let v = Geo::Inner(y = 1)\nend r",
                HiddenMemberError,
                "Geo::Inner",
            ),
        },
    ),
    "route-hiding": Scenario(
        modules={"tl": _TL},
        header=("import tl hiding Geo::Inner",),
        probes={
            "hidden-routed-value": rejected(
                "tl::Geo::Inner(y = 1)", HiddenMemberError, "tl::Geo::Inner"
            ),
            "hidden-routed-annotation": rejected(
                "fn(p: tl::Geo::Inner) => 1", HiddenMemberError, "tl::Geo::Inner"
            ),
            "unknown-routed-member": rejected("tl::Nope", UnknownMemberError, "tl::Nope"),
        },
    ),
    "type-arguments-per-segment": Scenario(
        modules={"lib": _GENERIC},
        header=("import lib::*", "def Outer::Inner::m[T](self) -> int = 1"),
        probes={
            "generic-segment-value": accepted(
                "Outer::Inner[int]::A(v = 1)", "record lib::Outer::Inner::A[int]\n  v: int"
            ),
            "generic-segment-annotation": accepted(
                "fn(p: Outer::Inner[int]::A) => p.v", "lib::Outer::Inner::A[int] -> int"
            ),
            "non-generic-segment": rejected(
                "fn(p: Outer[int]::Inner[int]::A) => 1", TypeArgumentsError, "Outer[int]"
            ),
            "wrong-arity-segment": rejected(
                "Outer::Inner[int, text]::A(v = 1)", TypeArgumentsError, "Inner[int, text]"
            ),
            "generic-segment-before-a-scope-path": rejected(
                "def Outer::Inner::sub::f() -> int = 1\n\nOuter::Inner[int]::sub::f()",
                TypeArgumentsError,
                "Inner[int]",
            ),
        },
    ),
    "type-arguments-select-an-inline-member": Scenario(
        header=(_NESTED_UNDER_GENERICS, "type Rows[A] = E[array[A]]"),
        probes={
            "inline-member-value": accepted("E[int]::A(a = 1)", "record E::A[int]\n  a: int"),
            "alias-inline-member-annotation": accepted(
                "fn(p: Rows[int]::A) => p.a", "E::A[array[int]] -> array[int]"
            ),
            "record-under-generic-record-value": rejected(
                "Box[int]::Inner(x = 1)", TypeArgumentsError, "Box[int]"
            ),
            "record-under-generic-record-annotation": rejected(
                "fn(p: Box[int]::Inner) => 1", TypeArgumentsError, "Box[int]"
            ),
            "record-under-generic-record-pattern": rejected(
                "case Box::Inner(x = 1) of\n  | Box[int]::Inner(x) => x",
                TypeArgumentsError,
                "Box[int]",
            ),
            "record-own-spelling": accepted("Box[int]::Box(v = 1)", "record Box[int]\n  v: int"),
            "record-own-spelling-pattern": accepted(
                "case Box(v = 1) of\n  | Box[int]::Box(v) => v", "int"
            ),
            "record-own-spelling-argument-mismatch": rejected(
                'Box[int]::Box(v = "s")', AglTypeError, '"s"', phase="typecheck"
            ),
            "record-own-spelling-pattern-on-another-application": rejected(
                'case Box(v = "s") of\n  | Box[int]::Box(v) => 1',
                AglTypeError,
                "Box[int]::Box(v)",
                phase="typecheck",
            ),
            "record-own-spelling-wrong-arity": rejected(
                "Box[int, int]::Box(v = 1)", TypeArgumentsError, "Box[int, int]"
            ),
            "non-generic-record-own-spelling": rejected(
                "Col::Red[int]::Red", TypeArgumentsError, "Red[int]"
            ),
            "record-under-generic-enum": rejected(
                "E[int]::Inner(x = 1)", TypeArgumentsError, "E[int]"
            ),
            "record-under-generic-alias": rejected(
                "Al[int]::Own(x = 1)", TypeArgumentsError, "Al[int]"
            ),
            "non-generic-owner-is": rejected(
                "Col::Red is Col[int]::Red", TypeArgumentsError, "Col[int]"
            ),
            "non-generic-owner-annotation": rejected(
                "fn(p: Col[int]::Red) => 1", TypeArgumentsError, "Col[int]"
            ),
            "too-many-arguments-cast": rejected(
                "fn(p: E[int]) => p as E[int, int]::B", TypeArgumentsError, "E[int, int]"
            ),
            "too-few-arguments-value": rejected(
                "Two[int]::L(l = 1)", TypeArgumentsError, "Two[int]"
            ),
        },
    ),
    "applied-owner-of-a-record-pattern": Scenario(
        header=(_NESTED_UNDER_GENERICS,),
        probes={
            "scoped-alias-owner": accepted(
                "scope q\n  type Rows[A] = E[array[A]]\n"
                "  let v = fn(r: E[array[int]]::A) => case r of\n"
                "    | Rows[int]::A(a) => a.size()\nend q\n\nq::v",
                "E::A[array[int]] -> int",
            ),
            "scoped-alias-owner-mismatch": rejected(
                "scope q\n  type Rows[A] = E[array[A]]\n"
                "  let v = fn(r: E[int]::A) => case r of\n"
                "    | Rows[int]::A(a) => 1\nend q",
                AglTypeError,
                "Rows[int]::A(a)",
                phase="typecheck",
            ),
        },
    ),
    "enum-references-a-member-through-a-use": Scenario(
        modules={"lib": _GEO_REGION},
        header=("import lib", "use lib::{Geo}", _REFERENCING_ENUM),
        probes=_REFERENCED_MEMBER_PROBES,
    ),
    "enum-references-a-nested-member-through-a-use": Scenario(
        modules={"lib": _GEO_TYPE},
        header=("import lib", "use lib::{Geo}", _REFERENCING_ENUM),
        probes=_REFERENCED_MEMBER_PROBES,
    ),
    "enum-in-a-region-references-a-member-through-its-use": Scenario(
        modules={"lib": _GEO_REGION},
        header=(
            "import lib",
            "scope S\n  use lib::{Geo}\n  enum E\n    | Geo::Point\n    | Other\nend S",
        ),
        probes={
            "referenced-member-through-the-enum": rejected(
                "S::E::Point(x = 1)", ReferencedMemberError, "S::E::Point"
            ),
            "value": accepted(
                "let e: S::E = lib::Geo::Point(x = 1)\ne",
                "enum S::E\n  | Point(x: int)\n  | Other",
            ),
        },
    ),
    "a-binding-reads-in-source-order": Scenario(
        modules={"tl": _TL, "tl2": _TL.replace("y: int", "z: int")},
        header=("import tl::*", "import tl2::*"),
        probes={
            "binding-before-a-declaration": rejected(
                "let p = fn(p: Geo::Inner::Nope) => 1\ndef Geo::Inner::m(self) -> int = 1",
                UnknownMemberError,
                "Geo::Inner::Nope",
            ),
            "declaration-before-a-binding": rejected(
                "def Geo::Inner::m(self) -> int = 1\nlet p = fn(p: Geo::Inner::Nope) => 1",
                AmbiguousQualificationError,
                "Geo::Inner",
            ),
        },
    ),
    "an-inline-entry-binds-at-the-module-root": Scenario(
        header=("var x = 1", "let y = 2"),
        probes={
            "anchored-var": accepted("::x", "int"),
            "anchored-var-assignment": accepted("::x := 3\nx", "int"),
            "anchored-let": accepted("::y", "int"),
            "anchored-let-assignment": rejected("::y := 3", AglTypeError, "::y", phase="typecheck"),
            "anchored-statement-binding": accepted("let z = true\n::z", "bool"),
            "anchored-from-a-region-declaring-the-name": accepted(
                "scope S\n  let y = true\n  def z() -> int = ::y\nend S\n\nS::z()", "int"
            ),
            "anchored-from-a-region-declaring-the-name-later": accepted(
                "scope S\n  def z() = ::y\n  let y = true\nend S\n\nS::z()", "int"
            ),
            "bare-beside-a-lambda-parameter-of-the-name": accepted(
                "def z() -> int = (fn(y: int) -> int => y)(5) + y\nz()", "int"
            ),
            "anchored-before-its-binding": rejected("::z\nlet z = true", UnknownMemberError, "::z"),
            "binding-claiming-a-declared-name": rejected(
                "def f() -> int = 1\nlet f = 2\nf", DuplicateDeclarationError, "let f = 2"
            ),
        },
    ),
    "receivers-read-the-full-path": Scenario(
        modules={"tl": _TL},
        header=("import tl::*", "record Geo\n  z: int"),
        probes={
            "imported-nested-under-own-type-name": accepted(
                "def Geo::Inner::m(self) -> int = self.y\ntl::Geo::Inner(y = 1).m()", "int"
            ),
            "own-type": accepted("def Geo::m(self) -> int = self.z\nGeo(z = 1).m()", "int"),
            "imported-nested-through-route": accepted(
                "def tl::Geo::Inner::m(self) -> int = self.y\ntl::Geo::Inner(y = 1).m()", "int"
            ),
        },
    ),
    "an-assignment-target-reads-like-a-value": Scenario(
        modules={"M": _COUNTED, "N": "scope Agent\n  var cnt: int = 0\nend Agent\n"},
        header=("import M::*", "import N::*"),
        probes={
            "ambiguous-qualified": rejected(
                "Agent::cnt := 2", AmbiguousQualificationError, "Agent::cnt"
            ),
            "ambiguous-bare": rejected(
                "scope Agent\n  def f() -> unit =\n    cnt := 2\nend Agent",
                AmbiguousQualificationError,
                "cnt",
            ),
            "immutable-qualified": rejected(
                "Agent::fixed := 2", AglTypeError, "Agent::fixed", phase="typecheck"
            ),
            "immutable-bare": rejected(
                "scope Agent\n  def f() -> unit =\n    fixed := 2\nend Agent",
                AglTypeError,
                "fixed",
                phase="typecheck",
            ),
            "missing-bare": rejected("nope := 2", AglScopeError, "nope"),
            "type-qualified": rejected("Agent::Mode := 2", AglScopeError, "Agent::Mode"),
            "routed-var": accepted("M::Agent::cnt := 2\nM::Agent::cnt", "int"),
        },
    ),
    "segment-type-arguments-need-the-decided-owner": Scenario(
        modules={"x/lib": _OPT, "y/two": "record Opt\n  w: int\n"},
        header=("import x/lib::*", "record Opt\n  z: int"),
        probes={
            "own-owner-annotation": rejected(
                "fn(p: Opt[int]::Som) => p.v", TypeArgumentsError, "Opt[int]"
            ),
            "own-owner-value": rejected("Opt[int]::Som(v = 1)", TypeArgumentsError, "Opt[int]"),
            "own-owner-pattern": rejected(
                "case lib::Opt::Som(v = 1) of\n  | Opt[int]::Som(v) => v",
                TypeArgumentsError,
                "Opt[int]",
            ),
            "member-path-without-arguments": accepted(
                "Opt::Som(v = 1)", "record x/lib::Opt::Som[int]\n  v: int"
            ),
            "routed-owner": accepted(
                "fn(p: lib::Opt[int]::Som) => p.v", "x/lib::Opt::Som[int] -> int"
            ),
        },
    ),
    "segment-type-arguments-on-an-ambiguous-owner": Scenario(
        modules={"x/lib": _OPT, "y/two": "record Opt\n  w: int\n"},
        header=("import x/lib::*", "import y/two::*"),
        probes={
            "annotation": rejected(
                "fn(p: Opt[int]::Som) => p.v", AmbiguousQualificationError, "Opt[int]"
            ),
            "value": rejected("Opt[int]::Som(v = 1)", AmbiguousQualificationError, "Opt[int]"),
            "member-path-without-arguments": accepted(
                "Opt::Som(v = 1)", "record x/lib::Opt::Som[int]\n  v: int"
            ),
        },
    ),
    "a-receiver-reads-its-whole-declaring-path": Scenario(
        modules={"x/lib": _NESTED_Y},
        header=(
            "import x/lib",
            "scope S\n  use lib::G::*\nend S",
            "record Z\n  c: int\nenum Tone\n  | Hi\n  | Lo",
        ),
        probes={
            "written-in-the-region": accepted(
                f"scope S\n  def Y::m(self) -> int = self.b\nend S\n{_NESTED_Y_CALL}", "int"
            ),
            "declaring-path-at-the-root": accepted(
                f"def S::Y::m(self) -> int = self.b\n{_NESTED_Y_CALL}", "int"
            ),
            "nested-regions": accepted(
                "scope S\n\n  scope Y\n    def m(self) -> int = self.b\n  end Y\nend S\n"
                + _NESTED_Y_CALL,
                "int",
            ),
            "path-region": accepted(
                f"scope S::Y\n  def m(self) -> int = self.b\nend S::Y\n{_NESTED_Y_CALL}", "int"
            ),
            "region-import-in-the-region": accepted(
                "scope T\n  import x/lib::*\n  def G::Y::m(self) -> int = self.b\nend T\n"
                + _NESTED_Y_CALL,
                "int",
            ),
            "region-import-at-the-root": accepted(
                "scope T\n  import x/lib::*\nend T\n\ndef T::G::Y::m(self) -> int = self.b\n"
                + _NESTED_Y_CALL,
                "int",
            ),
            "region-import-nested-regions": accepted(
                "scope T\n  import x/lib::*\n  scope G::Y\n    def m(self) -> int = self.b\n"
                f"  end G::Y\nend T\n{_NESTED_Y_CALL}",
                "int",
            ),
            "root-type-is-no-region-path": rejected(
                "scope S\n  def Z::m(self) -> int = self.c\nend S", AglScopeError, "self"
            ),
            "injected-member-at-the-root": accepted("def Hi::m(self) -> int = 1\nHi.m()", "int"),
            "injected-member-is-no-region-path": rejected(
                "scope S\n  def Hi::m(self) -> int = 1\nend S", AglScopeError, "self"
            ),
            "region-without-a-type": rejected(
                "scope Q\n  def m(self) -> int = 1\nend Q", AglScopeError, "self"
            ),
            "nested-region-without-a-type": rejected(
                "scope S\n\n  scope Q\n    def m(self) -> int = 1\n  end Q\nend S",
                AglScopeError,
                "self",
            ),
            "written-path-without-a-type": rejected(
                "def S::Q::m(self) -> int = 1", UnknownMemberError, "S::Q"
            ),
            "annotated-written-path-without-a-type": accepted(
                "def S::Q::m(self: Z) -> int = self.c\nS::Q::m(Z(c = 1))", "int"
            ),
        },
    ),
    "a-use-of-a-record-exposes-its-own-spelling-as-a-constructor-only": Scenario(
        modules={"e": "record Pt\n  x: int\nrecord Pt::In\n  y: int\n"},
        header=("import e",),
        probes={
            "constructor": accepted(
                _in_region("use e::Pt::*", "Pt(x = 1)"), "record e::Pt\n  x: int"
            ),
            "no-type": rejected(_in_region("use e::Pt::*", "fn(p: Pt) => p"), AglTypeError, "Pt"),
            "nested-type": accepted(
                _in_region("use e::Pt::*", "fn(p: In) => p"), "e::Pt::In -> e::Pt::In"
            ),
        },
    ),
    "a-tail-item-makes-its-qualifier-visible": Scenario(
        modules={"M": "scope Agent\n  def f() -> int = 1\n  def g() -> int = 2\nend Agent\n"},
        header=("import M",),
        probes={
            "selected": accepted(_in_region("use M::{Agent::f}", "Agent::f()"), "int"),
            "unselected": rejected(
                _in_region("use M::{Agent::f}", "Agent::g()"), UnknownMemberError, "Agent::g"
            ),
        },
    ),
    "an-injected-member-yields-only-to-a-record-the-use-exposes": Scenario(
        modules={
            "lib1": "scope X\n  enum E\n    | Red\n    | Blue\nend X\n",
            "lib2": "scope X\n  def Red() -> int = 1\nend X\n",
            "lib3": "record Pt\n  x: int\n\nscope X\n  type Red = Pt\nend X\n",
            "lib4": "scope X\n  record Red\n    x: int\nend X\n",
        },
        header=("import lib1::*",),
        probes={
            "beside-a-function": rejected(
                _in_region("import lib2::*\n  use X::*", "Red"), AmbiguousQualificationError, "Red"
            ),
            "beside-an-alias": rejected(
                _in_region("import lib3::*\n  use X::*", "Red"), AmbiguousConstructorError, "Red"
            ),
            "beside-a-record": accepted(
                _in_region("import lib4::*\n  use X::*", "Red"), "int -> lib4::X::Red"
            ),
            "another-member": accepted(
                _in_region("import lib4::*\n  use X::*", "Blue"), "record lib1::X::E::Blue"
            ),
        },
    ),
    "a-receiver-path-a-hiding-removed": Scenario(
        modules={"tl": _TL, "sc": "scope S\n  record T\n    y: int\nend S\n"},
        header=("import tl::* hiding Geo::Inner", "import sc::* hiding S::T"),
        probes={
            "nested-type": rejected(
                "def Geo::Inner::m(self) -> int = 1", HiddenMemberError, "Geo::Inner"
            ),
            "scoped-type": rejected("def S::T::m(self) -> int = 1", HiddenMemberError, "S::T"),
        },
    ),
    "receiver-in-own-region-of-type-name": Scenario(
        header=("enum Geo\n  | A\n  | B",),
        probes={
            "region-path-is-the-receiver": rejected(
                "scope r\n\n  scope Geo\n    def m(self) -> int = 1\n  end Geo\nend r",
                AglScopeError,
                "self",
            ),
            "region-prefix-is-part-of-the-path": rejected(
                "scope r\n  def helper() -> int = 1\nend r\n\ndef r::Geo::m(self) -> int = 1",
                UnknownMemberError,
                "r::Geo",
            ),
        },
    ),
    "a-use-reads-an-own-scope-declared-after-it": Scenario(
        modules={"M": _LATER_M},
        # One entry: a REPL use replaces an earlier one of the same target.
        header=("import M::*", "use Agent as A\nuse Agent::*", _LATER_OWN_SCOPE),
        probes={
            "alias-own": accepted("A::h()", "text"),
            "alias-imported": accepted("A::f()", "int"),
            "own": accepted("h()", "text"),
            "imported": accepted("f()", "int"),
        },
    ),
    "a-use-reads-own-bindings-declared-after-it": Scenario(
        modules={"M": _LATER_M},
        header=("import M::*", "use Agent::*", _LATER_OWN_BINDINGS),
        probes={
            "own-function-wins": accepted("f()", "text"),
            "own-binding-wins": accepted("cnt", "text"),
            "own-immutable-binding-wins-a-write": rejected(
                'cnt := "x"', AglTypeError, "cnt", phase="typecheck"
            ),
        },
    ),
    "a-use-reads-the-scope-a-nearer-use-exposes": Scenario(
        modules={"lib1": _X_A, "lib2": _X_B},
        header=(
            "import lib1",
            "import lib2",
            "use lib1::{X}",
            "scope S\n  use lib2::{X}\n  use X::*\n"
            "  def g() -> int = a()\n  def k() -> text = b()\nend S",
        ),
        probes={
            "from-the-enclosing-use": accepted("S::g()", "int"),
            "from-the-nearer-use": accepted("S::k()", "text"),
            "root-use-alone-at-the-root": accepted("X::a()", "int"),
            "nearer-use-stays-in-its-region": rejected("X::b()", UnknownMemberError, "X::b"),
        },
    ),
    "a-module-alias-combines-with-an-own-scope-of-its-name": Scenario(
        modules={"m": _ALIASED_MODULE},
        header=("import m as lib", _USE_OWN_LIB),
        probes={
            "routed": accepted("lib::f()", "int"),
            "routed-nested-scope": accepted("lib::Deep::d()", "int"),
            "own": accepted("lib::g()", "int"),
            "own-wins-its-path": accepted("lib::both()", "bool"),
            "used": accepted(_in_region("use lib::*", "f() + g()"), "int"),
            "used-own-wins-its-path": accepted(_in_region("use lib::*", "both()"), "bool"),
            "neither": rejected("lib::nope()", UnknownMemberError, "lib::nope"),
        },
    ),
    "a-middle-step-decides-a-qualified-chain": Scenario(
        modules={"M": "scope X\n  def f() -> bool = true\n  def k() -> bool = true\nend X\n"},
        header=("import M::*", "def X::f() -> int = 1", 'def S1::X::f() -> text = "m"'),
        probes={
            "middle-step": accepted(
                "scope S1::S2\n  let v = X::f()\nend S1::S2\n\nS1::S2::v", "text"
            ),
            "outer-step-when-the-middle-misses": accepted(
                "scope S1::S2\n  let v = X::k()\nend S1::S2\n\nS1::S2::v", "bool"
            ),
            "root-step": accepted("X::f()", "int"),
        },
    ),
    "a-branch-reads-a-constructor-pattern-name-like-a-bare-value": Scenario(
        modules={"M": _WRAPPED_COLOR, "N": _SIGNAL},
        header=("import M::*", "import N::*", "record Red\n  q: int"),
        probes={
            "own-at-the-root": accepted(_branch("Red(q = 1).q", region=False), "int"),
            "use-in-a-region": rejected(
                _branch("Red(q = 1).q", region=True), AglTypeError, "q = 1", phase="typecheck"
            ),
            "use-member-in-a-region": accepted(_branch("rank(Red)", region=True), "int"),
        },
    ),
    "a-branch-reading-an-ambiguous-constructor-pattern-name": Scenario(
        modules={"M": _WRAPPED_COLOR, "N": _SIGNAL},
        header=("import M::*", "import N::*"),
        probes={
            "ambiguous": rejected(
                _branch("rank(Red)", region=False),
                AglTypeError,
                "Red",
                phase="typecheck",
            ),
            "use-in-a-region": accepted(_branch("rank(Red)", region=True), "int"),
        },
    ),
    "a-field-access-object-reads-like-a-bare-value": Scenario(
        modules={"D": _BOUND_GEO},
        header=("import D", "record Geo\n  y: int"),
        probes={
            "use-in-a-region": accepted(
                "scope S\n  use D::{Geo}\n  def f() -> int = Geo.x\nend S\n\nS::f()", "int"
            ),
            "own-type-at-the-root": rejected("Geo.y", AglScopeError, "Geo"),
            "own-type-in-a-region": rejected(
                "scope S\n  def f() -> int = Geo.y\nend S", AglScopeError, "Geo"
            ),
        },
    ),
    "an-enum-injects-its-members-at-its-own-step": Scenario(
        header=(_SCOPED_COLOR, "def Red() -> int = 1"),
        probes={
            "in-its-scope": accepted(
                "scope S\n  let v = Red\nend S\n\nS::v", "record S::Color::Red"
            ),
            "in-a-nested-scope": accepted(
                "scope S::T\n  let v = Blue\nend S::T\n\nS::T::v", "record S::Color::Blue"
            ),
            "pattern-in-its-scope": accepted(
                "scope S\n  def f(c: Color) -> int =\n    case c of\n      | Red => 1\n"
                "      | Blue => 2\nend S\n\nS::f(S::Color::Red)",
                "int",
            ),
            "outside-its-scope": accepted("Red()", "int"),
            "outside-its-scope-other-member": rejected("Blue", AglScopeError, "Blue"),
            "no-member-path": rejected("S::Red", UnknownMemberError, "S::Red"),
            "own-record-at-its-step": accepted(
                "scope S\n  record Red\n    q: int\n  let v = Red(q = 1)\nend S\n\nS::v",
                "record S::Red\n  q: int",
            ),
        },
    ),
    "a-hiding-at-an-enclosing-step-holds-in-a-nested-region": Scenario(
        modules={"M": "scope Agent\n  def f() -> int = 1\n  def g() -> int = 2\nend Agent\n"},
        header=("import M",),
        probes={
            "bare": rejected(_in_nested("use M::Agent::* hiding f", "f()"), AglScopeError, "f"),
            "bare-kept": accepted(_in_nested("use M::Agent::* hiding f", "g()"), "int"),
            "qualified": rejected(
                _in_nested("use M::* hiding Agent::f", "Agent::f()"),
                HiddenMemberError,
                "Agent::f",
            ),
            "qualified-kept": accepted(_in_nested("use M::* hiding Agent::f", "Agent::g()"), "int"),
        },
    ),
}


class TestFullPathLookup:
    """Own-first full-path lookup in every position, file mode and every REPL grouping."""

    @pytest.mark.parametrize("scenario", scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", file_params(_SCENARIOS))
    def test_a_file_resolves_like_the_inline_entry(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        assert_file_resolves_like_inline_entry(tmp_path, scenario)


def test_scope_records_whether_a_spelling_selects_an_inline_member_or_a_declaration() -> None:
    """``E[int]::A`` selects enum ``E``'s inline member; ``E::Inner`` the record at that path."""
    source = (
        "enum E[T]\n  | A(a: T)\n  | B\nrecord E::Inner\n  x: int\n"
        "fn(p: E[int]::A, q: E::Inner) => q.x"
    )
    entry = resolve_inline_entry(source)
    names: list[NameT] = []
    walk(entry.program, lambda node: names.append(node) if isinstance(node, NameT) else None)
    member, nested = (
        entry.owner_declarations[node.qualifier.node_id]
        for node in names
        if node.qualifier is not None
    )

    assert member == OwnerMemberSelection((ENTRY_ID, (), "E"), "A")
    assert nested == DeclarationSelection((ENTRY_ID, ("E",), "Inner"))
