"""Parity tests for qualified constructor owners in expressions and patterns."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from agm.agl.modules.ids import ModuleId
from agm.agl.scope import AglScopeError
from agm.agl.scope.imports import ImportEnv, ModuleContribution
from agm.agl.scope.program import resolve_program
from agm.agl.syntax.nodes import QualifierChain, QualifierSegment
from agm.agl.syntax.spans import UNKNOWN_SOURCE, SourceSpan
from agm.agl.typecheck import AglTypeError
from agm.agl.typecheck.env import TypeEnvironment
from agm.agl.typecheck.program import check_program
from tests.agl.ir_harness import base_caps, make_graph_from_files
from tests.agl.module_graph import resolve_and_check_entry

Outcome = Literal["accepted", "scope", "typecheck"]


def _qualifier(*segments: str, member: str = "") -> QualifierChain:
    span = SourceSpan(1, 1, 1, 1, 0, 0, UNKNOWN_SOURCE)
    return QualifierChain(
        anchor=None,
        segments=tuple(
            QualifierSegment(name=name, type_args=None, span=span, node_id=index)
            for index, name in enumerate(segments)
        ),
        member=member,
        span=span,
        node_id=1,
    )


def _module_outcome(source: str) -> Outcome:
    try:
        resolve_and_check_entry(source, base_caps())
    except AglScopeError:
        return "scope"
    except AglTypeError:
        return "typecheck"
    return "accepted"


def _program_outcome(tmp_path: Path, modules: dict[str, str]) -> Outcome:
    try:
        resolved = resolve_program(make_graph_from_files(tmp_path, modules))
        check_program(resolved, base_caps())
    except AglScopeError:
        return "scope"
    except AglTypeError:
        return "typecheck"
    return "accepted"


@pytest.mark.parametrize(
    ("expression", "pattern", "expression_outcome", "pattern_outcome"),
    [
        (
            "enum Color\n  | Red\n::Color::Red",
            "enum Color\n  | Red\nlet value = Color::Red\ncase value of | ::Color::Red => 1",
            "accepted",
            "accepted",
        ),
        (
            "enum Color\n  | Red\nNope::Red",
            "enum Color\n  | Red\nlet value = Color::Red\ncase value of | Nope::Red => 1 | _ => 2",
            "scope",
            "typecheck",
        ),
        (
            "enum Color\n  | Red\n::Missing::Red",
            (
                "enum Color\n  | Red\nlet value = Color::Red\n"
                "case value of | ::Missing::Red => 1 | _ => 2"
            ),
            "scope",
            "typecheck",
        ),
        (
            "enum Color\n  | Red\nColor::Gone",
            (
                "enum Color\n  | Red\nlet value = Color::Red\n"
                "case value of | Color::Gone => 1 | _ => 2"
            ),
            "typecheck",
            "typecheck",
        ),
    ],
)
def test_expression_and_pattern_qualifier_verdicts_remain_in_parity(
    expression: str,
    pattern: str,
    expression_outcome: Outcome,
    pattern_outcome: Outcome,
) -> None:
    assert _module_outcome(expression) == expression_outcome
    assert _module_outcome(pattern) == pattern_outcome


def test_type_name_and_module_route_clash_stays_rejected_in_both_positions(
    tmp_path: Path,
) -> None:
    expression = {
        "entry": "import pkg/Foo\nenum Foo\n  | local\nFoo::local",
        "pkg/Foo": "def local() -> int = 1",
    }
    pattern = {
        "entry": (
            "import pkg/Foo\n"
            "enum Foo\n"
            "  | local\n"
            "let value = local\n"
            "case value of | Foo::local => 1"
        ),
        "pkg/Foo": "def local() -> int = 1",
    }

    assert _program_outcome(tmp_path / "expression", expression) == "scope"
    assert _program_outcome(tmp_path / "pattern", pattern) == "typecheck"


def test_unselected_qualified_owner_is_rejected_in_both_positions(tmp_path: Path) -> None:
    modules = {
        "entry": "import Pal using public\nPal::Secret::hidden",
        "Pal": "def public() -> int = 1\nenum Secret\n  | hidden",
    }
    pattern_modules = {
        **modules,
        "entry": (
            "import Pal using public\n"
            "let value = 1\n"
            "case value of | Pal::Secret::hidden => 1 | _ => 2"
        ),
    }

    assert _program_outcome(tmp_path / "expression", modules) == "scope"
    assert _program_outcome(tmp_path / "pattern", pattern_modules) == "typecheck"


def test_explicit_owner_matching_the_route_segment_is_rejected(tmp_path: Path) -> None:
    """A route segment sharing its spelling with a bogus explicit owner is rejected.

    ``pal::pal::Red`` spells the module route ``pal`` and an explicit (wrong)
    owner ``pal`` before ``Red``. The route segment happens to share its
    spelling with the explicit owner, but that coincidence must not make the
    route silently contribute the member itself as though no owner had been
    spelled — the qualifier is a bogus owner and must be rejected.
    """
    modules = {
        "entry": (
            "import pal\nlet value = pal::Color::Red\ncase value of | pal::pal::Red => 1 | _ => 2"
        ),
        "pal": "enum Color\n  | Red",
    }

    assert _program_outcome(tmp_path, modules) == "typecheck"


def test_correctly_spelled_module_and_owner_route_still_resolves(tmp_path: Path) -> None:
    """The correct spelling ``pal::Color::Red`` keeps resolving after the fix."""
    modules = {
        "entry": (
            "import pal\nlet value = pal::Color::Red\ncase value of | pal::Color::Red => 1 | _ => 2"
        ),
        "pal": "enum Color\n  | Red",
    }

    assert _program_outcome(tmp_path, modules) == "accepted"


def test_ambiguous_imported_owner_is_rejected_by_scope(tmp_path: Path) -> None:
    modules = {
        "entry": "import one/types\nimport two/types\nlet value = types::Color::Red\nvalue",
        "one/types": "enum Color\n  | Red",
        "two/types": "enum Color\n  | Red",
    }

    assert _program_outcome(tmp_path, modules) == "scope"


def test_typecheck_import_member_query_uses_the_shared_route_environment() -> None:
    module = ModuleId.from_path("pkg/types")
    import_env = ImportEnv(
        contributions={
            module: ModuleContribution(
                module=module,
                members={"Color": (module, "Color")},
                bare_names=frozenset(),
                path_enabled=True,
                aliases=frozenset(),
            )
        },
        unqualified={},
    )
    env = TypeEnvironment(import_env=import_env)

    assert env.has_qualified_import_member(_qualifier("types"), "Color")
    assert not env.has_qualified_import_member(_qualifier("types"), "Missing")
    assert not TypeEnvironment().has_qualified_import_member(_qualifier("types"), "Color")


def test_unselected_qualified_owner_is_rejected_by_typecheck_in_an_is_test(
    tmp_path: Path,
) -> None:
    modules = {
        "entry": (
            "import Pal using public\n"
            "enum Local\n"
            "  | ok\n"
            "let value = Local::ok\n"
            "value is Pal::Secret::hidden"
        ),
        "Pal": "def public() -> int = 1\nenum Secret\n  | hidden",
    }

    assert _program_outcome(tmp_path, modules) == "typecheck"


def test_ambiguous_qualified_owner_is_rejected_by_typecheck_in_an_is_test(
    tmp_path: Path,
) -> None:
    modules = {
        "entry": (
            "import one/types\n"
            "import two/types\n"
            "enum Local\n"
            "  | ok\n"
            "let value = Local::ok\n"
            "value is types::Color::Red"
        ),
        "one/types": "enum Color\n  | Red",
        "two/types": "enum Color\n  | Red",
    }

    assert _program_outcome(tmp_path, modules) == "typecheck"
