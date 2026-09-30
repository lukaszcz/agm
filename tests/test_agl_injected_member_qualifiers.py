"""A bare-injected enum member never qualifies anything.

An enum's members are injected bare as constructor values only: no
declaration has the full path ``Other::Other`` or ``Rec::Rec``, and a
constructor is no scope path, type, alias target or module route, so a chain
led by an injected spelling is an unknown qualifier in every position, over
every surface that injects it -- a facade ``use``, a module ``use``, an import
tail at the root or inside a region. Spelled bare in a type position, it
names no type.

Every probe is checked in file mode and in every legal REPL grouping of its
scenario's header (see :mod:`tests.agl.qualifier_support`), and a file
resolves exactly like the inline entry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.scope.symbols import UnknownQualifierError
from agm.agl.typecheck import AglTypeError
from tests.agl.qualifier_support import (
    Probe,
    Scenario,
    accepted,
    assert_file_resolves_like_inline_entry,
    assert_scenario_for_grouping,
    file_params,
    rejected,
    scenario_params,
)

_MODULES = {
    "other": "record Rec\n  x: int\n",
    "pk/lib": "import other\n\nenum E = other::Rec | Other\n",
}
_OTHER = "Other::Other"
_REC = "Rec::Rec"


def _unknown(text: str, chain: str) -> Probe:
    """*text* rejected in scope as an unknown qualifier spanning *chain*."""
    return rejected(text, UnknownQualifierError, chain)


def _no_type(text: str, parameter: str) -> Probe:
    """*text* rejected by typecheck: the annotation of *parameter* names no type."""
    return rejected(text, AglTypeError, parameter, phase="typecheck")


def _root_probes() -> dict[str, Probe]:
    """Every position spelling an injected member as a qualifier, at the root."""
    return {
        "injected-value": accepted("Other", "record pk/lib::E::Other"),
        "bare-annotation": _no_type("fn(x: Other) => 1", "x: Other"),
        "bare-reused-annotation": _no_type("fn(x: Rec) => 1", "x: Rec"),
        "value": _unknown("Other::Other", _OTHER),
        "constructor": _unknown("Rec::Rec(x = 1)", _REC),
        "annotation": _unknown("fn(x: Other::Other) => 1", _OTHER),
        "reused-annotation": _unknown("fn(x: Rec::Rec) => 1", _REC),
        "alias-target": _unknown("type A = Other::Other\nfn(x: A) => 1", _OTHER),
        "type-argument": _unknown("fn(x: array[Other::Other]) => 1", _OTHER),
        "pattern": _unknown(
            "let v: E = Other\ncase v of\n  | Other::Other => 1\n  | _ => 2", _OTHER
        ),
        "is": _unknown("let v: E = Other\nv is Rec::Rec", _REC),
        "cast": _unknown("let v: E = Other\nv as? Other::Other", _OTHER),
    }


def _in_region(*lines: str) -> str:
    """*lines* inside ``scope s``, which imports ``pk/lib``'s members itself."""
    body = "\n".join(f"  {line}" for line in lines)
    return f"scope s\n  import pk/lib::*\n{body}\nend s"


def _region_probes() -> dict[str, Probe]:
    """Every position spelling an injected member as a qualifier, inside the importing region."""
    return {
        "injected-value": accepted(
            _in_region("let q = Other") + "\n\ns::q", "record pk/lib::E::Other"
        ),
        "bare-annotation": _no_type(_in_region("def f(x: Other) -> int = 1"), "x: Other"),
        "bare-reused-annotation": _no_type(_in_region("def f(x: Rec) -> int = 1"), "x: Rec"),
        "value": _unknown(_in_region("let q = Other::Other"), _OTHER),
        "constructor": _unknown(_in_region("let q = Rec::Rec(x = 1)"), _REC),
        "annotation": _unknown(_in_region("def f(x: Other::Other) -> int = 1"), _OTHER),
        "reused-annotation": _unknown(_in_region("def f(x: Rec::Rec) -> int = 1"), _REC),
        "alias-target": _unknown(_in_region("type A = Other::Other"), _OTHER),
        "type-argument": _unknown(_in_region("def f(x: array[Other::Other]) -> int = 1"), _OTHER),
        "pattern": _unknown(
            _in_region(
                "def f(v: E) -> int =", "  case v of", "    | Other::Other => 1", "    | _ => 2"
            ),
            _OTHER,
        ),
        "is": _unknown(_in_region("def f(v: E) -> bool = v is Rec::Rec"), _REC),
        "cast": _unknown(_in_region("def f(v: E) = v as? Other::Other"), _OTHER),
    }


_SCENARIOS = {
    "facade-use": Scenario(
        modules=_MODULES, header=("import pk/* as F", "use F::*"), probes=_root_probes()
    ),
    "module-use": Scenario(
        modules=_MODULES, header=("import pk/lib", "use pk/lib::*"), probes=_root_probes()
    ),
    "import-tail": Scenario(modules=_MODULES, header=("import pk/lib::*",), probes=_root_probes()),
    "region-import-tail": Scenario(modules=_MODULES, header=(), probes=_region_probes()),
}


class TestInjectedMemberQualifiers:
    """An injected member leading a chain, in every position and over every surface."""

    @pytest.mark.parametrize(("scenario", "sizes"), scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(
        self, tmp_path: Path, scenario: Scenario, sizes: tuple[int, ...]
    ) -> None:
        assert_scenario_for_grouping(tmp_path, scenario, sizes)

    @pytest.mark.parametrize("scenario", file_params(_SCENARIOS))
    def test_a_file_resolves_like_the_inline_entry(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        assert_file_resolves_like_inline_entry(tmp_path, scenario)
