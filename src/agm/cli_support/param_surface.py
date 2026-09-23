"""Pure spelling and precedence table for module parameters."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from agm.agl.runtime.types import ParamBindingInfo, ProgramDeclInfo
from agm.agl.zones import ParamZone
from agm.cli_support.program_options import ProjectedOption, param_spellings, project_option
from agm.config.qualified_keys import route_table_paths

__all__ = ["ParamSurface", "ParamSurfaceEntry", "build_param_surface"]


@dataclass(frozen=True, slots=True)
class ParamSurfaceEntry:
    """One module parameter's spellings after precedence has been applied."""

    param: ParamBindingInfo
    spellings: tuple[str, ...]
    option_spellings: tuple[str, ...]
    positive_option_spellings: tuple[str, ...]
    negative_option_spellings: tuple[str, ...]
    short_option: str | None
    section: str
    hidden: bool


@dataclass(frozen=True, slots=True)
class ParamSurface:
    """The resolving and ambiguous spellings of one program's module parameters."""

    entries: tuple[ParamSurfaceEntry, ...]
    ambiguous: Mapping[str, tuple[ParamBindingInfo, ...]]
    ambiguous_options: Mapping[str, tuple[ParamBindingInfo, ...]]


@dataclass(frozen=True, slots=True)
class _ParamRoutes:
    """One module parameter's long-option names and their CLI projections.

    Derived once per parameter and threaded through every level, since both
    the claim levels and the resolved entry read the same routes.
    """

    param: ParamBindingInfo
    spellings: tuple[str, ...]
    projections: tuple[ProjectedOption, ...]


_Claimant = ParamBindingInfo | None
_Claims = dict[str, tuple[_Claimant, ...]]


def _append_claim(claims: dict[str, list[_Claimant]], spelling: str, claimant: _Claimant) -> None:
    claims.setdefault(spelling, []).append(claimant)


def _freeze_claims(claims: dict[str, list[_Claimant]]) -> _Claims:
    return {spelling: tuple(claimants) for spelling, claimants in claims.items()}


def _param_routes(param: ParamBindingInfo) -> _ParamRoutes:
    """Project *param*'s logical long-option names, shortest route first."""
    spellings = [param.cli.name]
    if not param.module.is_entry:
        spellings.extend(
            ".".join((*path, param.cli.name))
            for path in route_table_paths(param.module.segments, param.scope_path)
            if all("/" not in segment for segment in path)
        )
    unique_spellings: dict[str, None] = {}
    for spelling in spellings:
        unique_spellings[spelling] = None
    names = tuple(unique_spellings)
    return _ParamRoutes(
        param=param,
        spellings=names,
        projections=tuple(project_option(name, param.type) for name in names),
    )


def _host_long_claims(reserved_flags: Collection[str]) -> _Claims:
    """Project host long flags into the bare-name level they reserve."""
    claims: dict[str, list[_Claimant]] = {}
    for flag in reserved_flags:
        if not flag.startswith("--"):
            continue
        name = flag[2:]
        _append_claim(claims, name, None)
        if name.startswith("no-"):
            _append_claim(claims, name[3:], None)
    return _freeze_claims(claims)


def _host_option_claims(reserved_flags: Collection[str]) -> _Claims:
    """Return the exact option flags the host has already claimed."""
    return {flag: (None,) for flag in reserved_flags}


def _signature_long_claims(program: ProgramDeclInfo) -> _Claims:
    claims: dict[str, list[_Claimant]] = {}
    for parameter in program.parameters:
        _append_claim(claims, parameter.cli.name, None)
    return _freeze_claims(claims)


def _signature_option_claims(program: ProgramDeclInfo) -> _Claims:
    claims: dict[str, list[_Claimant]] = {}
    for parameter in program.parameters:
        if parameter.kind is ParamZone.POSITIONAL_ONLY:
            continue
        projected = project_option(parameter.cli.name, parameter.type)
        for flag in param_spellings(parameter, projected):
            _append_claim(claims, flag, None)
    return _freeze_claims(claims)


def _param_long_claims(routes: Iterable[_ParamRoutes]) -> _Claims:
    claims: dict[str, list[_Claimant]] = {}
    for route in routes:
        for spelling in route.spellings:
            _append_claim(claims, spelling, route.param)
    return _freeze_claims(claims)


def _param_option_claims(routes: Iterable[_ParamRoutes]) -> _Claims:
    claims: dict[str, list[_Claimant]] = {}
    for route in routes:
        for spelling in param_spellings(route.param, *route.projections):
            _append_claim(claims, spelling, route.param)
    return _freeze_claims(claims)


def _resolve_claims(
    levels: tuple[_Claims, ...],
) -> tuple[dict[str, ParamBindingInfo | None], dict[str, tuple[ParamBindingInfo, ...]]]:
    """Resolve claims lexically, retaining same-level parameter ambiguities."""
    winners: dict[str, ParamBindingInfo | None] = {}
    ambiguous: dict[str, tuple[ParamBindingInfo, ...]] = {}
    claimed: set[str] = set()
    for level in levels:
        for spelling, claimants in level.items():
            if spelling in claimed:
                continue
            claimed.add(spelling)
            if len(claimants) == 1:
                winners[spelling] = claimants[0]
                continue
            params = tuple(param for param in claimants if param is not None)
            if params:
                ambiguous[spelling] = params
    return winners, ambiguous


def build_param_surface(
    reserved_flags: Collection[str],
    program: ProgramDeclInfo,
    params: Iterable[ParamBindingInfo],
) -> ParamSurface:
    """Build the resolving parameter spellings for *program*.

    The host and program signature claim names before module parameters.  The
    selected program's module is its own level; every other supplied module
    shares the final import level, so only imported peers can make a spelling
    ambiguous.
    """
    routes = tuple(_param_routes(param) for param in params)
    own = tuple(route for route in routes if route.param.module == program.module)
    imported = tuple(route for route in routes if route.param.module != program.module)

    host_long = _host_long_claims(reserved_flags)
    signature_long = _signature_long_claims(program)
    own_long = _param_long_claims(own)
    imported_long = _param_long_claims(imported)
    long_winners, ambiguous = _resolve_claims((host_long, signature_long, own_long, imported_long))

    host_options = _host_option_claims(reserved_flags)
    signature_options = _signature_option_claims(program)
    own_options = _param_option_claims(own)
    imported_options = _param_option_claims(imported)
    option_winners, ambiguous_options = _resolve_claims(
        (host_options, signature_options, own_options, imported_options)
    )

    entries = tuple(_entry(route, long_winners, option_winners) for route in (*own, *imported))
    return ParamSurface(
        entries=entries,
        ambiguous=MappingProxyType(ambiguous),
        ambiguous_options=MappingProxyType(ambiguous_options),
    )


def _entry(
    route: _ParamRoutes,
    long_winners: Mapping[str, ParamBindingInfo | None],
    option_winners: Mapping[str, ParamBindingInfo | None],
) -> ParamSurfaceEntry:
    """Keep the spellings *route*'s parameter won, split by the polarity each fills."""
    param = route.param
    spellings = tuple(
        spelling for spelling in route.spellings if long_winners.get(spelling) == param
    )
    positive_option_spellings: list[str] = []
    negative_option_spellings: list[str] = []
    option_spellings: list[str] = []
    for projected in route.projections:
        if option_winners.get(projected.flag) == param:
            positive_option_spellings.append(projected.flag)
            option_spellings.append(projected.flag)
        negative = projected.negative_flag
        if negative is not None and option_winners.get(negative) == param:
            negative_option_spellings.append(negative)
            option_spellings.append(negative)
    short = param.cli.short
    short_option = (
        None if short is None or option_winners.get(f"-{short}") != param else f"-{short}"
    )
    if short_option is not None:
        option_spellings.append(short_option)
    return ParamSurfaceEntry(
        param=param,
        spellings=spellings,
        option_spellings=tuple(option_spellings),
        positive_option_spellings=tuple(positive_option_spellings),
        negative_option_spellings=tuple(negative_option_spellings),
        short_option=short_option,
        section=param.module.display(),
        hidden=param.cli.hidden,
    )
