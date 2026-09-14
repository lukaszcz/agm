"""Tests for the pure module-parameter spelling surface."""

from __future__ import annotations

from dataclasses import replace

from agm.agl.attributes import ProgramOptionSpec
from agm.agl.modules.ids import ENTRY_ID, ModuleId
from agm.agl.runtime.types import ParamBindingInfo, ProgramDeclInfo, ProgramParamInfo
from agm.agl.semantics.types import BoolType, IntType, TextType, Type
from agm.agl.syntax.spans import SourceSpan
from agm.agl.zones import ParamZone
from agm.cli_support.param_surface import ParamSurface, ParamSurfaceEntry, build_param_surface
from agm.cli_support.program_options import EXEC_RESERVED_FLAGS, REGISTERED_RESERVED_FLAGS

_SPAN = SourceSpan(1, 1, 1, 2, 0, 1)


def _option(name: str, *, short: str | None = None, hidden: bool = False) -> ProgramOptionSpec:
    return ProgramOptionSpec(
        name=name,
        short=short,
        env=None,
        metavar=None,
        hidden=hidden,
        doc=None,
    )


def _param(
    module: str,
    name: str,
    type_: Type = BoolType(),
    *,
    scope: tuple[str, ...] = (),
    external: str | None = None,
    short: str | None = None,
    hidden: bool = False,
) -> ParamBindingInfo:
    return ParamBindingInfo(
        module=ModuleId.from_path(module),
        scope_path=scope,
        name=name,
        node_id=1,
        span=_SPAN,
        type=type_,
        mutable=False,
        cli=_option(name if external is None else external, short=short, hidden=hidden),
        doc=None,
    )


def _signature(
    name: str,
    *,
    kind: ParamZone = ParamZone.NAMED_ONLY,
    short: str | None = None,
) -> ProgramParamInfo:
    return ProgramParamInfo(
        name=name,
        kind=kind,
        type=TextType(),
        has_default=False,
        span=_SPAN,
        cli=_option(name, short=short),
    )


def _program(*parameters: ProgramParamInfo) -> ProgramDeclInfo:
    return ProgramDeclInfo(
        module=ModuleId.from_path("app/main"),
        scope_path=(),
        name="run",
        node_id=1,
        span=_SPAN,
        parameters=parameters,
        is_entry=True,
        doc=None,
        closure=(ModuleId.from_path("app/main"),),
    )


def _entry(surface: ParamSurface, param: ParamBindingInfo) -> ParamSurfaceEntry:
    return next(entry for entry in surface.entries if entry.param == param)


def test_levels_resolve_from_host_to_signature_to_own_then_imports() -> None:
    own_verbose = _param("app/main", "verbose")
    own_limit = _param("app/main", "limit")
    own_theme = _param("app/main", "theme")
    imported_verbose = _param("lib/logging", "verbose")
    imported_theme = _param("lib/theme", "theme")
    surface = build_param_surface(
        frozenset({"--verbose"}),
        _program(_signature("limit")),
        (own_verbose, own_limit, own_theme, imported_verbose, imported_theme),
    )

    assert _entry(surface, own_verbose).spellings == ("main.verbose", "app.main.verbose")
    assert _entry(surface, own_limit).spellings == ("main.limit", "app.main.limit")
    assert _entry(surface, own_theme).spellings[0] == "theme"
    assert _entry(surface, imported_verbose).spellings == (
        "logging.verbose",
        "lib.logging.verbose",
    )
    assert _entry(surface, imported_theme).spellings == ("theme.theme", "lib.theme.theme")
    assert surface.shadowed_bare == frozenset({"verbose", "limit"})


def test_import_bare_collision_is_ambiguous_but_qualified_forms_resolve() -> None:
    first = _param("first/logging", "verbose")
    second = _param("second/logging", "verbose")
    surface = build_param_surface(frozenset(), _program(), (first, second))

    assert surface.ambiguous["verbose"] == (first, second)
    assert "verbose" not in _entry(surface, first).spellings
    assert _entry(surface, first).spellings == ("first.logging.verbose",)
    assert _entry(surface, second).spellings == ("second.logging.verbose",)
    assert surface.ambiguous["logging.verbose"] == (first, second)
    assert "--verbose" in surface.ambiguous_options


def test_own_module_claims_a_bare_name_before_an_import() -> None:
    own = _param("app/main", "verbose")
    imported = _param("lib/logging", "verbose")
    surface = build_param_surface(frozenset(), _program(), (imported, own))

    assert _entry(surface, own).spellings[0] == "verbose"
    assert "verbose" not in _entry(surface, imported).spellings
    assert "verbose" not in surface.ambiguous


def test_root_and_scope_params_get_each_module_suffix_route_shortest_first() -> None:
    root = _param("A/logging", "verbose")
    scoped = _param("A/logging", "trace", scope=("debug",))
    surface = build_param_surface(frozenset(), _program(), (root, scoped))

    assert _entry(surface, root).spellings == (
        "verbose",
        "logging.verbose",
        "A.logging.verbose",
    )
    assert _entry(surface, scoped).spellings == (
        "trace",
        "logging.debug.trace",
        "A.logging.debug.trace",
    )


def test_option_name_and_short_spelling_are_resolved_on_the_same_level() -> None:
    param = _param("A/logging", "trace", external="trace-level", short="t")
    surface = build_param_surface(frozenset(), _program(), (param,))
    entry = _entry(surface, param)

    assert entry.section == "A/logging"
    assert entry.spellings == (
        "trace-level",
        "logging.trace-level",
        "A.logging.trace-level",
    )
    assert entry.option_spellings == (
        "--trace-level",
        "--no-trace-level",
        "--logging.trace-level",
        "--no-logging.trace-level",
        "--A.logging.trace-level",
        "--no-A.logging.trace-level",
        "-t",
    )


def test_hidden_param_remains_resolvable_and_is_marked_hidden() -> None:
    param = _param("A/logging", "quiet", hidden=True)
    surface = build_param_surface(frozenset(), _program(), (param,))

    entry = _entry(surface, param)
    assert entry.hidden is True
    assert entry.spellings[0] == "quiet"


def test_exec_and_registered_host_surfaces_shadow_different_bare_names() -> None:
    log = _param("A/logging", "log")
    call_depth = _param("A/logging", "max-call-depth", IntType())
    exec_surface = build_param_surface(EXEC_RESERVED_FLAGS, _program(), (log, call_depth))
    registered_surface = build_param_surface(
        REGISTERED_RESERVED_FLAGS, _program(), (log, call_depth)
    )

    assert exec_surface.shadowed_bare == frozenset({"log", "max-call-depth"})
    assert registered_surface.shadowed_bare == frozenset()
    assert "log" not in _entry(exec_surface, log).spellings
    assert _entry(registered_surface, log).spellings[0] == "log"
    assert "max-call-depth" not in _entry(exec_surface, call_depth).spellings
    assert _entry(registered_surface, call_depth).spellings[0] == "max-call-depth"


def test_only_negatable_projections_contribute_negation_spellings() -> None:
    flag = _param("A/logging", "verbose", BoolType())
    text = _param("A/logging", "label", TextType())
    surface = build_param_surface(frozenset(), _program(), (flag, text))

    assert "--no-verbose" in _entry(surface, flag).option_spellings
    text_spellings = _entry(surface, text).option_spellings
    assert all("--no-label" not in spelling for spelling in text_spellings)


def test_entries_are_ordered_by_the_shortest_resolving_spelling() -> None:
    param = _param("A/B/logging", "verbose", scope=("debug",))
    surface = build_param_surface(frozenset(), _program(), (param,))

    assert _entry(surface, param).spellings == (
        "verbose",
        "logging.debug.verbose",
        "B.logging.debug.verbose",
        "A.B.logging.debug.verbose",
    )


def test_anonymous_entry_param_has_only_its_cli_bare_spelling() -> None:
    param = replace(_param("app/main", "verbose"), module=ENTRY_ID)
    surface = build_param_surface(frozenset(), _program(), (param,))

    entry = _entry(surface, param)
    assert entry.spellings == ("verbose",)
    assert entry.section == "<entry>"


def test_signature_short_and_positional_parameters_claim_only_their_valid_flags() -> None:
    param = _param("app/main", "verbose", short="v")
    positional = _param("app/main", "positional")
    signature = (
        _signature("quiet", short="v"),
        _signature("positional", kind=ParamZone.POSITIONAL_ONLY),
    )
    surface = build_param_surface(frozenset(), _program(*signature), (param, positional))

    assert "-v" not in _entry(surface, param).option_spellings
    assert "positional" in surface.shadowed_bare
