"""``hiding`` removes a declaration, a scope prefix or an alias from one contribution.

Every contribution kind (import tail, region import, import route, ``use``
glob, item, rename and alias, re-export, alias target) reaches the same
verdict: the removed spelling is a hidden-member error when nothing else
supplies it, a bare spelling is simply unknown, and a miss that removed nothing
the spelling would reach is not hidden.

Every probe is checked in the file part and in every legal REPL grouping of its
scenario's header (see :mod:`tests.agl.qualifier_support`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.diagnostics import HiddenMemberError
from agm.agl.scope.symbols import AglScopeError, UnknownMemberError, UnknownQualifierError
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


class TestHidingMechanism:
    """Hidden verdicts and selection across every contribution kind."""

    @pytest.mark.parametrize("scenario", scenario_params(_SCENARIOS))
    def test_file_and_every_repl_grouping_agree(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.xfail(
        strict=True,
        reason="a hidden spelling written through a use alias reads as an unknown member",
    )
    @pytest.mark.parametrize("scenario", scenario_params(_ALIAS_READS_HIDING))
    def test_use_alias_reads_hiding(self, tmp_path: Path, scenario: Scenario) -> None:
        assert_scenario(tmp_path, scenario)

    @pytest.mark.xfail(
        strict=True,
        reason="a chain through a hidden function reads as hidden instead of an unknown qualifier",
    )
    @pytest.mark.parametrize("scenario", scenario_params(_HIDDEN_FUNCTION_QUALIFIER))
    def test_chain_through_hidden_function_is_unknown_qualifier(
        self, tmp_path: Path, scenario: Scenario
    ) -> None:
        assert_scenario(tmp_path, scenario)
