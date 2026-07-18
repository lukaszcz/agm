"""Focused scope and typecheck coverage for slash namespace contributions."""

from __future__ import annotations

from pathlib import Path

import pytest

from agm.agl.matchcompile.diagnostics import qualified_owner_name
from agm.agl.modules.ids import ModuleId
from agm.agl.modules.loader import load_graph
from agm.agl.modules.roots import RootSet
from agm.agl.scope.program import resolve_program
from agm.agl.scope.symbols import AglScopeError
from agm.agl.semantics.types import EnumOwnerFormKind
from agm.agl.syntax.spans import UNKNOWN_SOURCE, SourceSpan
from agm.agl.syntax.types import Qualifier
from agm.agl.typecheck import AglTypeError
from agm.agl.typecheck.program import check_program
from tests.agl.ir_harness import base_caps, make_graph_from_files, write_module_file


def _make_graph_without_prelude(tmp_path: Path, modules: dict[str, str]) -> object:
    root = tmp_path / "root"
    root.mkdir()
    for module_path, source in modules.items():
        if module_path != "entry":
            write_module_file(root, module_path, source)
    return load_graph(
        modules["entry"], entry_path=None, roots=RootSet(frozenset({root})), default_stdlib=False
    )


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


def test_is_test_type_and_module_constructor_member_collision_is_ambiguous(
    tmp_path: Path,
) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import support/config\n"
                "enum config | On\n"
                "let flag = ::config::On\n"
                "flag is config::On"
            ),
            "support/config": "def On() -> int = 1",
        },
    )

    resolved = resolve_program(graph)
    with pytest.raises(AglTypeError) as exc_info:
        check_program(resolved, base_caps())

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
                "import support/config\nenum config | On\nlet flag = config::On\nflag is config::On"
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
    rendered = qualified_owner_name("Flag", form.module_qualifier, anchored=form.qualifier_anchored)
    assert rendered == "/remote/config::Flag"


def test_qualified_enum_owner_form_rejects_a_non_type_member(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import remote/config\n0",
            "remote/config": "def Flag() -> int = 1",
        },
    )
    checked = check_program(resolve_program(graph), base_caps())
    qualifier = Qualifier(
        segments=("remote", "config"),
        anchored=False,
        span=SourceSpan(1, 1, 1, 1, 0, 0, UNKNOWN_SOURCE),
        node_id=0,
    )

    assert (
        checked.modules[graph.entry_id].type_env.resolve_enum_owner_form(
            EnumOwnerFormKind.QUALIFIED_IMPORT,
            "Flag",
            qualifier,
        )
        is None
    )


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


def test_enum_owner_forms_exclude_ambiguous_suffix_routes(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": (
                "import one/config\n"
                "import two/config\n"
                "let flag = one/config::Flag::On\n"
                "case flag of | one/config::Flag::On => 1 | _ => 2"
            ),
            "one/config": "enum Flag | On | Off",
            "two/config": "enum Flag | On | Off",
        },
    )

    checked = check_program(resolve_program(graph), base_caps())
    forms = checked.modules[graph.entry_id].type_env.enum_owner_forms()

    assert not any(form.module_qualifier == ("config",) for form in forms)
    assert {
        form.module_qualifier
        for form in forms
        if form.owner_name == "Flag" and form.module_qualifier is not None
    } >= {("one", "config"), ("two", "config")}


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


def test_spec_suffixes_anchor_and_two_line_bare_full_idiom(tmp_path: Path) -> None:
    """Suffixes filter by member, anchors select exactly, and imports union."""
    graph = _make_graph_without_prelude(
        tmp_path,
        {
            "entry": (
                "import std/config\n"
                "import std/list/config\n"
                "import extra/config hiding retries\n"
                "import utils/api\n"
                "import utils/api using bare\n"
                "let a = config::retries()\n"
                "let b = list/config::opt()\n"
                "let c = /std/config::opt()\n"
                "let d = bare()\n"
                "let e = api::full()\n"
                "e"
            ),
            "std/config": "def retries() -> int = 1\ndef opt() -> int = 2",
            "std/list/config": "def opt() -> int = 3",
            "extra/config": "def retries() -> int = 4\ndef opt() -> int = 5",
            "utils/api": "def bare() -> int = 6\ndef full() -> int = 7",
        },
    )

    resolved = resolve_program(graph)
    entry = resolved.modules[graph.entry_id]
    assert entry.import_env.unqualified["bare"] == frozenset(
        {(ModuleId.from_path("utils/api"), "bare")}
    )


def test_hiding_repairs_a_suffix_ambiguity_and_new_import_makes_it_loud(tmp_path: Path) -> None:
    modules = {
        "one/config": "def opt() -> int = 1",
        "two/config": "def opt() -> int = 2",
    }
    repaired = make_graph_from_files(
        tmp_path,
        {"entry": "import one/config\nimport two/config hiding opt\nconfig::opt()", **modules},
    )
    assert resolve_program(repaired).entry_id == repaired.entry_id

    ambiguous = make_graph_from_files(
        tmp_path,
        {"entry": "import one/config\nimport two/config\nconfig::opt()", **modules},
    )
    with pytest.raises(AglScopeError, match="ambiguous"):
        resolve_program(ambiguous)


def test_wildcard_alias_is_a_member_filtered_facade(tmp_path: Path) -> None:
    graph = make_graph_from_files(
        tmp_path,
        {
            "entry": "import facade/* as api\nlet first = api::first()\napi::second()",
            "facade/one": "def first() -> int = 1",
            "facade/two": "def second() -> int = 2",
        },
    )

    assert resolve_program(graph).entry_id == graph.entry_id
