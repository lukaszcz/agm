"""``hiding`` removes a declaration, a scope prefix or an alias from one contribution.

Every contribution kind (import tail, region import, import route, ``use``
glob, item, rename and alias, re-export, alias target) reaches the same
verdict: the removed spelling is a hidden-member error when nothing else
supplies it, a bare spelling is simply unknown, and a miss that removed nothing
the spelling would reach is not hidden.

Every probe runs inline; verdict representatives also cover file and REPL
entry boundaries through :mod:`tests.agl.qualifier_support`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from agm.agl.diagnostics import AglTypeError, HiddenMemberError
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import (
    AglScopeError,
    AmbiguousConstructorError,
    DuplicateDeclarationError,
    UnknownMemberError,
    UnknownQualifierError,
)
from tests.agl.ir_harness import make_inline_graph_from_files
from tests.agl.qualifier_support import (
    Scenario,
    accepted,
    assert_scenario,
    rejected,
    scenario_params,
)

_LIB = (
    "scope S\n  def x() -> int = 1\n  def y() -> int = 2\nend S\n\n"
    "def f() -> int = 3\n\n"
    "record Box\n  n: int\n\n"
    "def Box::j() -> int = 4\n\n"
    "type HBox = Box\n"
)
_OTHER = 'def f() -> text = "o"\n\nscope S\n  def x() -> text = "s"\nend S\n'
_MID = "import lib\nexport lib hiding S::x\n"
_TYPES_LIB = (
    "scope S\n  def x() -> int = 1\nend S\n\n"
    "record Box\n  n: int\n\nenum Color\n  | Red\n  | Blue\n"
)


def _withheld(hidden: str) -> dict[str, str]:
    """Modules: ``mid`` re-exports ``lib`` withholding *hidden*; ``top`` re-exports ``mid``."""
    return {
        "lib": _TYPES_LIB,
        "mid": f"import lib\nexport lib hiding {hidden}\n",
        "top": "import mid\nexport mid\n",
    }


def _withheld_scenarios(hidden: str) -> dict[str, Scenario]:
    """Rows where an export ``hiding`` withholds the unexported type or scope *hidden*.

    Whatever *hidden* names stays hidden as a qualifier through every spelling.
    """
    return {
        f"re-export-withholding-{hidden}-through-a-glob-import": Scenario(
            modules=_withheld(hidden),
            header=("import mid::*",),
            probes={
                "use-glob": rejected(f"use {hidden}::*", HiddenMemberError, f"use {hidden}::*"),
                "use-alias": rejected(
                    f"use {hidden} as X", HiddenMemberError, f"use {hidden} as X"
                ),
                "member": rejected(f"{hidden}::nope()", HiddenMemberError, f"{hidden}::nope"),
            },
        ),
        f"re-export-withholding-{hidden}-through-a-route": Scenario(
            modules=_withheld(hidden),
            header=("import mid",),
            probes={
                "use-alias": rejected(
                    f"use mid::{hidden} as X", HiddenMemberError, f"use mid::{hidden} as X"
                ),
                "member": rejected(
                    f"mid::{hidden}::nope()", HiddenMemberError, f"mid::{hidden}::nope"
                ),
            },
        ),
        f"double-re-export-withholding-{hidden}": Scenario(
            modules=_withheld(hidden),
            header=("import top::*",),
            probes={
                "use-glob": rejected(f"use {hidden}::*", HiddenMemberError, f"use {hidden}::*")
            },
        ),
    }


_SCENARIOS = {
    "import-tail-hides-a-declaration": Scenario(
        modules={"lib": _LIB},
        header=("import lib::* hiding S::x",),
        probes={
            "hidden": rejected("S::x()", HiddenMemberError, "S::x"),
            "route": rejected("lib::S::x()", HiddenMemberError, "lib::S::x"),
            "anchored": rejected("/lib::S::x()", HiddenMemberError, "/lib::S::x"),
            "sibling": accepted("S::y()", "int"),
            "never-declared": rejected("S::nope()", UnknownMemberError, "S::nope"),
        },
    ),
    "import-tail-hides-a-bare-declaration": Scenario(
        modules={"lib": _LIB, "other": _OTHER},
        header=("import lib::* hiding f", "import other::*"),
        probes={
            "other-selected": accepted("f()", "text"),
            "route-hidden": rejected("lib::f()", HiddenMemberError, "lib::f"),
        },
    ),
    "import-tail-hides-a-bare-declaration-alone": Scenario(
        modules={"lib": _LIB},
        header=("import lib::* hiding f",),
        probes={
            "bare-unknown": rejected("f()", AglScopeError, "f"),
            "route-hidden": rejected("lib::f()", HiddenMemberError, "lib::f"),
            "kept": accepted("lib::S::y()", "int"),
        },
    ),
    "import-tail-hides-a-scope-prefix": Scenario(
        modules={"lib": _LIB},
        header=("import lib::* hiding S",),
        probes={
            "member": rejected("S::x()", HiddenMemberError, "S::x"),
            "route-member": rejected("lib::S::y()", HiddenMemberError, "lib::S::y"),
            "beneath-never-declared": rejected("S::nope::x()", HiddenMemberError, "S::nope::x"),
            "kept": accepted("f()", "int"),
        },
    ),
    "import-tail-hides-an-alias-and-what-it-reaches": Scenario(
        modules={"lib": _LIB},
        header=("import lib::* hiding HBox",),
        probes={
            "alias-member": rejected("HBox::j()", HiddenMemberError, "HBox::j"),
            "target-member": rejected("Box::j()", HiddenMemberError, "Box::j"),
            "kept": accepted("f()", "int"),
        },
    ),
    "import-tail-hides-a-member-through-an-alias": Scenario(
        modules={"lib": _LIB},
        header=("import lib::* hiding HBox::j",),
        probes={
            "member": rejected("Box::j()", HiddenMemberError, "Box::j"),
            "through-alias": rejected("HBox::j()", HiddenMemberError, "HBox::j"),
            "route": rejected("lib::Box::j()", HiddenMemberError, "lib::Box::j"),
            "kept": accepted("Box(n = 1).n", "int"),
        },
    ),
    "import-route-hides-a-declaration": Scenario(
        modules={"lib": _LIB},
        header=("import lib hiding S::x",),
        probes={
            "route": rejected("lib::S::x()", HiddenMemberError, "lib::S::x"),
            "anchored": rejected("/lib::S::x()", HiddenMemberError, "/lib::S::x"),
            "sibling": accepted("lib::S::y()", "int"),
            "bare-unknown": rejected("S::x()", UnknownQualifierError, "S::x"),
        },
    ),
    "region-import-hides-a-declaration": Scenario(
        modules={"lib": _LIB},
        header=(),
        probes={
            "hidden": rejected(
                "scope r\n  import lib::* hiding S::x\n  def v() -> int = S::x()\nend r",
                HiddenMemberError,
                "S::x",
            ),
            "sibling": accepted(
                "scope r\n  import lib::* hiding S::x\n  def v() -> int = S::y()\nend r\n\nr::v()",
                "int",
            ),
        },
    ),
    "region-import-hiding-leaves-the-outside-alone": Scenario(
        modules={"lib": _LIB},
        header=("import lib::*",),
        probes={
            "outside": accepted("scope r\n  import lib::* hiding S::x\nend r\n\nS::x()", "int"),
        },
    ),
    "region-import-hides-a-scope-prefix": Scenario(
        modules={"lib": _LIB},
        header=(),
        probes={
            "hidden": rejected(
                "scope r\n  import lib::* hiding S\n  def v() -> int = S::x()\nend r",
                HiddenMemberError,
                "S::x",
            ),
            "kept": accepted(
                "scope r\n  import lib::* hiding S\n  def v() -> int = f()\nend r\n\nr::v()",
                "int",
            ),
        },
    ),
    "use-glob-hides-a-declaration": Scenario(
        modules={"lib": _LIB},
        header=("import lib\nuse lib::* hiding S::x",),
        probes={
            "hidden": rejected("S::x()", HiddenMemberError, "S::x"),
            "sibling": accepted("S::y()", "int"),
            "bare": accepted("f()", "int"),
        },
    ),
    "use-glob-hides-a-scope-prefix": Scenario(
        modules={"lib": _LIB},
        header=("import lib\nuse lib::* hiding S",),
        probes={
            "member": rejected("S::x()", HiddenMemberError, "S::x"),
            "kept": accepted("f()", "int"),
        },
    ),
    "use-glob-hides-an-alias": Scenario(
        modules={"lib": _LIB},
        header=("import lib\nuse lib::* hiding HBox",),
        probes={
            "alias-member": rejected("HBox::j()", HiddenMemberError, "HBox::j"),
            "target-member": rejected("Box::j()", HiddenMemberError, "Box::j"),
            "kept": accepted("f()", "int"),
        },
    ),
    "use-glob-hides-a-member-through-an-alias": Scenario(
        modules={"lib": _LIB},
        header=("import lib\nuse lib::* hiding HBox::j",),
        probes={
            "member": rejected("Box::j()", HiddenMemberError, "Box::j"),
            "through-alias": rejected("HBox::j()", HiddenMemberError, "HBox::j"),
        },
    ),
    "use-target-through-a-hidden-scope": Scenario(
        modules={"lib": _LIB},
        header=("import lib::* hiding S",),
        probes={"target": rejected("use S::*", HiddenMemberError, "use S::*")},
    ),
    "use-glob-beneath-a-hidden-member": Scenario(
        modules={"lib": _LIB},
        header=("import lib::* hiding S::x\nuse S::*",),
        probes={
            "bare-hidden": rejected("x()", AglScopeError, "x"),
            "bare-kept": accepted("y()", "int"),
        },
    ),
    "another-contribution-supplies-the-qualified-path": Scenario(
        modules={"lib": _LIB},
        header=("import lib::* hiding S::x\nimport lib as L",),
        probes={"through-route": accepted("L::S::x()", "int")},
    ),
    "another-import-selected-for-the-hidden-path": Scenario(
        modules={"lib": _LIB, "other": _OTHER},
        header=("import lib::* hiding S::x\nimport other::*",),
        probes={"other-selected": accepted("S::x()", "text")},
    ),
    "bare-re-export-importer-hides-a-scope-prefix": Scenario(
        modules={"lib": _LIB, "mid": "import lib\nexport lib hiding S\n"},
        header=("import mid::*",),
        probes={
            "hidden": rejected("S::x()", HiddenMemberError, "S::x"),
            "kept": accepted("f()", "int"),
        },
    ),
    "use-item-and-rename-beside-an-import-hiding": Scenario(
        modules={"lib": _LIB},
        header=("import lib::* hiding f\nimport lib as L", "use L::{f as h}"),
        probes={
            "renamed": accepted("h()", "int"),
            "source": accepted("f()", "int"),
        },
    ),
    "use-item-supplies-what-an-import-hides": Scenario(
        modules={"lib": _LIB},
        header=("import lib::* hiding S::x\nimport lib as L", "use L::{S::x}"),
        probes={
            "qualified": accepted("S::x()", "int"),
        },
    ),
    "use-alias-of-a-scope-reads-the-same-hiding": Scenario(
        modules={"lib": _LIB},
        header=("import lib::* hiding S::y", "use S as X"),
        probes={
            "alias-kept": accepted("X::x()", "int"),
            "direct-hidden": rejected("S::y()", HiddenMemberError, "S::y"),
        },
    ),
    "use-alias-of-a-module-keeps-what-the-import-keeps": Scenario(
        modules={"lib": _LIB},
        header=("import lib hiding S::x", "use lib as L"),
        probes={"kept": accepted("L::S::y()", "int")},
    ),
    "re-export-hiding-a-declaration": Scenario(
        modules={"lib": _LIB, "mid": _MID},
        header=("import mid",),
        probes={
            "hidden": rejected("mid::S::x()", HiddenMemberError, "mid::S::x"),
            "sibling": accepted("mid::S::y()", "int"),
        },
    ),
    "re-export-hiding-an-alias": Scenario(
        modules={"lib": _LIB, "mid": "import lib\nexport lib hiding HBox\n"},
        header=("import mid::*",),
        probes={
            "alias-member": rejected("HBox::j()", HiddenMemberError, "HBox::j"),
            "target-member": rejected("Box::j()", HiddenMemberError, "Box::j"),
            "kept": accepted("f()", "int"),
        },
    ),
}

_METHOD_LIB = (
    "record Box\n  n: int\n\ndef Box::j(self) -> int = 4\n\ntype HBox = Box\n\n"
    "def mk() -> Box = Box(n = 1)\n"
)

_ALIAS_READS_HIDING = {
    "import-tail-hiding-is-read-through-a-use-alias": Scenario(
        modules={"lib": _LIB},
        header=("import lib::* hiding S::x", "use S as X"),
        probes={"alias-hidden": rejected("X::x()", HiddenMemberError, "X::x")},
    ),
    "use-glob-hiding-is-read-through-a-use-alias": Scenario(
        modules={"lib": _LIB},
        header=("import lib\nuse lib::* hiding S::x", "use S as X"),
        probes={"alias-hidden": rejected("X::x()", HiddenMemberError, "X::x")},
    ),
    "module-use-alias-reads-an-import-hiding": Scenario(
        modules={"lib": _LIB},
        header=("import lib hiding S::x", "use lib as L"),
        probes={"alias-hidden": rejected("L::S::x()", HiddenMemberError, "L::S::x")},
    ),
    "region-use-alias-reads-an-import-hiding": Scenario(
        modules={"lib": _LIB},
        header=(),
        probes={
            "alias-hidden": rejected(
                "scope r\n  import lib::* hiding S::x\n  use S as X\n"
                "  def v() -> int = X::x()\nend r",
                HiddenMemberError,
                "X::x",
            )
        },
    ),
}

_HIDDEN_FUNCTION_QUALIFIER = {
    "hidden-function-is-not-a-qualifier": Scenario(
        modules={"lib": _LIB},
        header=("import lib::* hiding S::x",),
        probes={"beneath": rejected("S::x::z()", UnknownQualifierError, "S::x::z")},
    ),
}


_WITHHELD = {
    **_withheld_scenarios("Color"),
    **_withheld_scenarios("Box"),
    **_withheld_scenarios("S"),
}

_NESTED_LIB = (
    "scope S\n\n  scope T\n    def x() -> int = 1\n  end T\n\n  def y() -> int = 2\nend S\n\n"
    "def f() -> int = 3\n"
)


def _two_exports(hidden: str) -> dict[str, str]:
    """Modules: ``p/one`` re-exports ``lib`` withholding *hidden*; ``p/two`` withholds nothing."""
    return {
        "lib": _LIB,
        "p/one": f"import lib\nexport lib hiding {hidden}\n",
        "p/two": "import lib\nexport lib\n",
    }


_REACHED_ONE_WAY = {
    "wildcard-reaches-one-export-withholding-a-declaration": Scenario(
        modules=_two_exports("f"),
        header=("import p/*",),
        probes={
            "route": rejected("one::f()", HiddenMemberError, "one::f"),
            "path-route": rejected("p/one::f()", HiddenMemberError, "p/one::f"),
            "other-export": accepted("two::f()", "int"),
        },
    ),
    "wildcard-reaches-one-export-withholding-a-scope": Scenario(
        modules=_two_exports("S"),
        header=("import p/*",),
        probes={
            "member": rejected("one::S::x()", HiddenMemberError, "one::S::x"),
            "use-alias": rejected("use one::S as X", HiddenMemberError, "use one::S as X"),
            "other-export": accepted("two::S::x()", "int"),
        },
    ),
    "export-beside-a-region-export-withholds-in-the-root-only": Scenario(
        modules={
            "lib": _LIB,
            "mid": "import lib\nexport lib hiding S, f\n\nscope R\n  export lib\nend R\n",
        },
        header=("import mid::*",),
        probes={
            "bare": rejected("f()", AglScopeError, "f"),
            "member": rejected("S::x()", HiddenMemberError, "S::x"),
            "use-alias": rejected("use S as X", HiddenMemberError, "use S as X"),
            "region-kept": accepted("R::S::x()", "int"),
        },
    ),
    "export-beside-a-region-export-withholds-in-the-root-only-through-a-route": Scenario(
        modules={
            "lib": _LIB,
            "mid": "import lib\nexport lib hiding S, f\n\nscope R\n  export lib\nend R\n",
        },
        header=("import mid",),
        probes={
            "bare": rejected("mid::f()", HiddenMemberError, "mid::f"),
            "member": rejected("mid::S::x()", HiddenMemberError, "mid::S::x"),
            "region-kept": accepted("mid::R::S::x()", "int"),
        },
    ),
    "region-export-withholds-beside-a-root-export": Scenario(
        modules={
            "lib": _LIB,
            "mid": "import lib\nexport lib\n\nscope R\n  export lib hiding S\nend R\n",
        },
        header=("import mid::*",),
        probes={
            "region-member": rejected("R::S::x()", HiddenMemberError, "R::S::x"),
            "root-kept": accepted("S::x()", "int"),
            "region-sibling": accepted("R::f()", "int"),
        },
    ),
    "renamed-declaration-withheld-beside-its-unrenamed-export": Scenario(
        modules={
            "lib": _NESTED_LIB,
            "mid": "import lib\nexport lib::{S::T as U}\nexport lib::{S}\n",
            "top": "import mid\nexport mid hiding U\n",
        },
        header=("import top::*",),
        probes={
            "use-glob": rejected("use U::*", HiddenMemberError, "use U::*"),
        },
    ),
    "renamed-declaration-withheld-beside-its-unrenamed-export-through-a-route": Scenario(
        modules={
            "lib": _NESTED_LIB,
            "mid": "import lib\nexport lib::{S::T as U}\nexport lib::{S}\n",
            "top": "import mid\nexport mid hiding U\n",
        },
        header=("import top",),
        probes={
            "use-alias": rejected("use top::U as X", HiddenMemberError, "use top::U as X"),
        },
    ),
}

_REGION_RE_EXPORTS = {
    "region-export-withholds-a-scope-with-what-lies-beneath": Scenario(
        modules={
            "lib": _NESTED_LIB,
            "mid": "import lib\n\nscope R\n  export lib hiding S\nend R\n",
        },
        header=("import mid::*",),
        probes={
            "use-alias": rejected("use R::S as X", HiddenMemberError, "use R::S as X"),
            "use-glob": rejected("use R::S::*", HiddenMemberError, "use R::S::*"),
            "member": rejected("R::S::nope()", HiddenMemberError, "R::S::nope"),
            "kept": accepted("R::f()", "int"),
        },
    ),
    "region-export-withholds-a-nested-scope-and-keeps-its-owner": Scenario(
        modules={
            "lib": _NESTED_LIB,
            "mid": "import lib\n\nscope R\n  export lib hiding S::T\nend R\n",
        },
        header=("import mid::*",),
        probes={
            "member": rejected("R::S::T::nope()", HiddenMemberError, "R::S::T::nope"),
            "use-alias": rejected("use R::S::T as X", HiddenMemberError, "use R::S::T as X"),
            "owner-kept": accepted("R::S::y()", "int"),
        },
    ),
}

_PREFIX_VERDICTS = {
    "hidden-scope-another-import-declares": Scenario(
        modules={"lib": _LIB, "other": _OTHER},
        header=("import lib::* hiding S\nimport other::*",),
        probes={
            "unknown-member": rejected("S::nope()", UnknownMemberError, "S::nope"),
            "other-selected": accepted("S::x()", "text"),
        },
    ),
    "region-hides-a-scope-an-outer-step-reaches": Scenario(
        modules={"lib": _LIB},
        header=("import lib::*",),
        probes={
            "unknown-member": rejected(
                "scope r\n  import lib::* hiding S\n  def v() -> int = S::nope()\nend r",
                UnknownMemberError,
                "S::nope",
            ),
            "outer-selected": accepted(
                "scope r\n  import lib::* hiding S\n  def v() -> int = S::x()\nend r\n\nr::v()",
                "int",
            ),
        },
    ),
}

_HIDDEN_METHODS = {
    "import-tail-hides-an-alias-of-the-receiver": Scenario(
        modules={"lib": _METHOD_LIB},
        header=("import lib::* hiding HBox",),
        probes={
            "method": rejected("mk().j()", AglTypeError, "mk().j", phase="typecheck"),
            "field": accepted("mk().n", "int"),
        },
    ),
    "import-tail-hides-a-method-through-an-alias": Scenario(
        modules={"lib": _METHOD_LIB},
        header=("import lib::* hiding HBox::j",),
        probes={"method": rejected("mk().j()", AglTypeError, "mk().j", phase="typecheck")},
    ),
    "import-tail-keeps-the-method": Scenario(
        modules={"lib": _METHOD_LIB},
        header=("import lib::*",),
        probes={"method": accepted("mk().j()", "int")},
    ),
}


class TestHidingMechanism:
    """Hidden verdicts and selection across every contribution kind."""

    @pytest.mark.parametrize("scenario", scenario_params(_SCENARIOS))
    def test_semantics_and_entry_boundaries(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_ALIAS_READS_HIDING))
    def test_use_alias_reads_hiding(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_HIDDEN_FUNCTION_QUALIFIER))
    def test_chain_through_hidden_function_is_unknown_qualifier(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_WITHHELD))
    def test_re_export_withholding_a_type_or_scope_keeps_it_hidden(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize(
        "scenario", scenario_params({**_REACHED_ONE_WAY, **_REGION_RE_EXPORTS})
    )
    def test_hiding_belongs_to_the_way_a_declaration_is_reached(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_PREFIX_VERDICTS))
    def test_prefix_is_hidden_only_when_no_step_reaches_it(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.parametrize("scenario", scenario_params(_HIDDEN_METHODS))
    def test_method_visibility_follows_hiding(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)


def test_ambiguity_repair_ignores_a_hidden_owner_of_the_same_name(tmp_path: Path) -> None:
    """A hidden ``Color`` does not make ``a``'s ``Color`` need its module route to be spelled."""
    modules = {
        "a": "enum Color\n  | Red\n  | Blue\n",
        "d": "enum Light\n  | Red\n",
        "b": "enum Color\n  | Green\n",
        "entry": "import a::*\nimport d::*\nimport b::* hiding Color\nRed",
    }
    with pytest.raises(AmbiguousConstructorError) as caught:
        resolve_program(make_inline_graph_from_files(tmp_path, modules))
    assert caught.value.repair == "Color::Red"


_BAD_HIDING_LIB = "record Box\n  n: int\n\ndef Box::k() -> int = 5\n\ntype HBox = Box\n"
_TWO_BAD_ITEMS = {
    "lib": _BAD_HIDING_LIB,
    "mid": "import lib\nexport lib hiding HBox::k\n",
    "entry": "import mid::* hiding HBox::nope\nimport mid::* hiding HBox::zip\n1",
}
_FIRST_BAD_ITEM_PROBE = """
import sys
from pathlib import Path
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import UnknownMemberError
from tests.agl.ir_harness import make_inline_graph_from_files

modules = {modules!r}
try:
    resolve_program(make_inline_graph_from_files(Path(sys.argv[1]), modules))
except UnknownMemberError as error:
    print(error.span.start_line)
"""


@pytest.mark.parametrize("seed", ["0", "1", "2", "3"])
def test_first_bad_hiding_item_is_reported_whatever_the_hash_seed(
    tmp_path: Path, seed: str
) -> None:
    """Two bad ``hiding`` items, one through an export-withheld way: the first in source order."""
    result = subprocess.run(
        [sys.executable, "-c", _FIRST_BAD_ITEM_PROBE.format(modules=_TWO_BAD_ITEMS), str(tmp_path)],
        env={**os.environ, "PYTHONHASHSEED": seed},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "1"


def test_duplicate_declaration_is_reported_before_a_bad_hiding_item(tmp_path: Path) -> None:
    modules = {
        "lib": _BAD_HIDING_LIB,
        "entry": ("import lib::* hiding HBox::nope\ndef f() -> int = 1\ndef f() -> int = 2\n1"),
    }
    with pytest.raises(DuplicateDeclarationError):
        resolve_program(make_inline_graph_from_files(tmp_path, modules))
