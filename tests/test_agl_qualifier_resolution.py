"""Parity tests for qualified constructor owners in expressions and patterns."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from agm.agl.modules.ids import ModuleId
from agm.agl.parser import parse_program
from agm.agl.scope import AglScopeError, resolve_module
from agm.agl.scope.imports import (
    ImportEnv,
    ModuleContribution,
    OwnerSelfModule,
    QualResolutionUnknownQualifier,
)
from agm.agl.scope.program import resolve_program
from agm.agl.semantics.types import EnumType
from agm.agl.syntax.spans import UNKNOWN_SOURCE, SourceSpan
from agm.agl.syntax.types import Qualifier
from agm.agl.typecheck import AglTypeError, check_module
from agm.agl.typecheck.env import TypeEnvironment
from agm.agl.typecheck.program import check_program
from tests.agl.ir_harness import base_caps, make_graph_from_files

Outcome = Literal["accepted", "scope", "typecheck"]


def _qualifier(*segments: str) -> Qualifier:
    return Qualifier(
        segments=segments,
        span=SourceSpan(1, 1, 1, 1, 0, 0, UNKNOWN_SOURCE),
        node_id=1,
    )


def _module_outcome(source: str) -> Outcome:
    try:
        resolved = resolve_module(parse_program(source))
        check_module(resolved, base_caps())
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


def test_private_qualified_owner_is_rejected_in_both_positions(tmp_path: Path) -> None:
    modules = {
        "entry": "import Pal\nPal::Secret::hidden",
        "Pal": "def public() -> int = 1\nprivate enum Secret\n  | hidden",
    }
    pattern_modules = {
        **modules,
        "entry": ("import Pal\nlet value = 1\ncase value of | Pal::Secret::hidden => 1 | _ => 2"),
    }

    assert _program_outcome(tmp_path / "expression", modules) == "scope"
    assert _program_outcome(tmp_path / "pattern", pattern_modules) == "typecheck"


def test_ambiguous_imported_owner_is_rejected_by_scope(tmp_path: Path) -> None:
    modules = {
        "entry": "import one/types\nimport two/types\nlet value = types::Color::Red\nvalue",
        "one/types": "enum Color\n  | Red",
        "two/types": "enum Color\n  | Red",
    }

    assert _program_outcome(tmp_path, modules) == "scope"


def test_typecheck_owner_verdicts_cover_self_and_no_import_routes() -> None:
    env = TypeEnvironment()
    env.register_type("Color", EnumType("Color"))

    assert env.resolve_constructor_owner(_qualifier(), "Color", "Red") == OwnerSelfModule("Color")
    assert env.resolve_constructor_owner(
        _qualifier("missing"), "missing", "Red"
    ) == QualResolutionUnknownQualifier(("missing",))


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


def test_private_qualified_owner_is_rejected_by_typecheck_in_an_is_test(tmp_path: Path) -> None:
    modules = {
        "entry": (
            "import Pal\nenum Local\n  | ok\nlet value = Local::ok\nvalue is Pal::Secret::hidden"
        ),
        "Pal": "def public() -> int = 1\nprivate enum Secret\n  | hidden",
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
