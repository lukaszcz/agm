"""Focused scope and typecheck coverage for slash namespace contributions."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.matchcompile.diagnostics import qualified_owner_name
from agm.agl.modules.ids import ModuleId
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import AglScopeError
from agm.agl.semantics.types import EnumOwnerFormKind
from agm.agl.syntax.spans import UNKNOWN_SOURCE, SourceSpan
from agm.agl.syntax.types import Qualifier
from agm.agl.typecheck import AglTypeError
from agm.agl.typecheck.program import check_program
from tests.agl.ir_harness import base_caps, make_graph_from_files


def test_scope_resolves_suffix_anchor_and_using_contributions(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import services/primary/config\n"
                "import services/secondary/config\n"
                "import services/primary/config using primary\n"
                "let first = /services/primary/config::primary()\n"
                "let second = config::secondary()\n"
                "let third = primary()\n"
                "third"
            ),
            "services/primary/config": "def primary() -> int = 1",
            "services/secondary/config": "def secondary() -> int = 2",
        },
    )

    resolved = resolve_program(graph)
    entry = resolved.modules[graph.entry_id]
    primary = ModuleId.from_path("services/primary/config")
    secondary = ModuleId.from_path("services/secondary/config")

    assert entry.import_env.unqualified["primary"] == frozenset({(primary, "primary")})
    resolved_modules = {ref.module_id for ref in entry.resolved.resolution.values()}
    assert primary in resolved_modules
    assert secondary in resolved_modules


def test_scope_rejects_an_ambiguous_suffix_at_the_use_site(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import one/config\nimport two/config\nconfig::shared()",
            "one/config": "def shared() -> int = 1",
            "two/config": "def shared() -> int = 2",
        },
    )

    with pytest.raises(AglScopeError, match="ambiguous") as exc_info:
        resolve_program(graph)

    diagnostic = str(exc_info.value)
    for repair in ("hiding", "longer suffix", "/-anchored", "as"):
        assert repair in diagnostic


def test_typecheck_routes_qualified_types_patterns_and_is_tests(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import services/flags/config\n"
                "let flag: /services/flags/config::Flag = /services/flags/config::Flag::On\n"
                "let result = case flag of\n"
                "  | /services/flags/config::Flag::On => 1\n"
                "  | /services/flags/config::Flag::Off => 2\n"
                "flag is /services/flags/config::Flag::On"
            ),
            "services/flags/config": "enum Flag | On | Off",
        },
    )

    checked = check_program(resolve_program(graph), base_caps())

    assert ModuleId.from_path("services/flags/config") in checked.modules


def test_type_qualifier_beats_route_without_the_requested_member(tmp_path: Path) -> None:
    """A shared route is irrelevant until it contributes the constructor member."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import support/config\n"
                "enum config | On | Off\n"
                "let flag = config::On\n"
                "let result = case flag of\n"
                "  | config::On => 1\n"
                "  | config::Off => 2\n"
                "flag is config::On"
            ),
            "support/config": "def unrelated() -> int = 1",
        },
    )

    assert check_program(resolve_program(graph), base_caps()).entry_id == graph.entry_id


def test_type_and_module_constructor_member_collision_is_ambiguous(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import support/config\nenum config | On\nconfig::On",
            "support/config": "def On() -> int = 1",
        },
    )

    with pytest.raises(AglScopeError) as exc_info:
        resolve_program(graph)

    diagnostic = str(exc_info.value)
    assert "both a type name and a module route" in diagnostic
    for repair in ("hiding", "longer suffix", "/-anchored", "as"):
        assert repair in diagnostic


def test_is_test_does_not_treat_an_imported_enum_owner_as_its_variant_route(
    tmp_path: Path,
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import support/config\n"
                "enum config | On\n"
                "let flag = config::On\n"
                "flag is config::On"
            ),
            "support/config": "enum config | On",
        },
    )

    assert check_program(resolve_program(graph), base_caps()).entry_id == graph.entry_id


@pytest.mark.parametrize(
    ("source", "modules", "expected"),
    [
        (
            "enum Flag | On | Off\nlet flag = Flag::On\nflag is unknown::Flag::On",
            {},
            "Unknown module qualifier",
        ),
        (
            "import remote/config hiding Flag\nenum Flag | On | Off\n"
            "let flag = Flag::On\n"
            "let result = case flag of\n"
            "  | config::Flag::On => 1\n"
            "  | _ => 2\n"
            "result",
            {"remote/config": "enum Flag | On | Off"},
            "not accessible",
        ),
        (
            "import one/config\nimport two/config\nenum Local | On\n"
            "let flag = Local::On\nflag is config::Flag::On",
            {"one/config": "enum Flag | On", "two/config": "enum Flag | On"},
            "ambiguous",
        ),
    ],
)
def test_qualified_enum_patterns_and_is_tests_keep_resolution_verdicts(
    tmp_path: Path,
    source: str,
    modules: dict[str, str],
    expected: str,
) -> None:
    graph = make_graph_from_files(tmp_path, {"entry": source, **modules})

    with pytest.raises(AglTypeError) as exc_info:
        check_program(resolve_program(graph), base_caps())

    assert expected in str(exc_info.value)


def test_qualified_private_type_keeps_the_private_diagnostic(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import remote/config\nlet value: /remote/config::Flag = 1\nvalue",
            "remote/config": "private enum Flag | On",
        },
    )

    with pytest.raises(AglTypeError) as exc_info:
        check_program(resolve_program(graph), base_caps())

    assert "private" in str(exc_info.value)


@pytest.mark.parametrize(
    "entry",
    [
        "import remote/config\nrecord Wrapper\n  flag: /remote/config::Flag\nWrapper",
        "import remote/config\ndef inspect(flag: /remote/config::Flag) -> int = 1\ninspect",
    ],
)
def test_qualified_private_type_keeps_private_diagnostic_during_prepasses(
    tmp_path: Path, entry: str
) -> None:
    """Type-body and signature pre-passes retain private declaration metadata."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": entry,
            "remote/config": "private enum Flag | On",
        },
    )

    with pytest.raises(AglTypeError, match="private"):
        check_program(resolve_program(graph), base_caps())


def test_anchored_enum_owner_form_preserves_its_route(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import remote/config\nlet flag = /remote/config::Flag::On\nflag",
            "remote/config": "enum Flag | On",
        },
    )
    checked = check_program(resolve_program(graph), base_caps())
    env = checked.modules[graph.entry_id].type_env
    qualifier = Qualifier(
        segments=("remote", "config"),
        anchored=True,
        span=SourceSpan(1, 1, 1, 1, 0, 0, UNKNOWN_SOURCE),
        node_id=0,
    )

    form = env.resolve_enum_owner_form(
        kind=EnumOwnerFormKind.QUALIFIED_IMPORT,
        owner_name="Flag",
        module_qualifier=qualifier,
    )

    assert form is not None
    assert form.qualifier_anchored is True
    rendered = qualified_owner_name(
        "Flag", form.module_qualifier, anchored=form.qualifier_anchored
    )
    assert rendered == "/remote/config::Flag"


def test_pattern_and_is_filter_type_module_routes_by_the_referenced_variant(tmp_path: Path) -> None:
    """A module's enum owner does not route its variants directly."""
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import support/config\n"
                "enum config | On | Off\n"
                "let flag = config::On\n"
                "let result = case flag of\n"
                "  | config::On => 1\n"
                "  | config::Off => 2\n"
                "flag is config::On"
            ),
            "support/config": "enum config | External",
        },
    )

    assert check_program(resolve_program(graph), base_caps()).entry_id == graph.entry_id


def test_anchored_constructor_route_never_falls_back_to_a_local_type(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import remote/config as C\nenum C | On\n/C::On",
            "remote/config": "enum C | External",
        },
    )

    with pytest.raises(AglScopeError, match="No module imported"):
        resolve_program(graph)
