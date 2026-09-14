"""Pure spelling and precedence table for module parameters."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from agm.agl.runtime.types import ParamBindingInfo, ProgramDeclInfo
from agm.agl.zones import ParamZone
from agm.cli_support.program_options import project_option
from agm.config.qualified_keys import route_table_paths

__all__ = ["ParamSurface", "ParamSurfaceEntry", "build_param_surface"]


@dataclass(frozen=True, slots=True)
class ParamSurfaceEntry:
    """One module parameter's spellings after precedence has been applied."""

    param: ParamBindingInfo
    spellings: tuple[str, ...]
    option_spellings: tuple[str, ...]
    section: str
    hidden: bool


@dataclass(frozen=True, slots=True)
class ParamSurface:
    """The resolving and ambiguous spellings of one program's module parameters."""

    entries: tuple[ParamSurfaceEntry, ...]
    ambiguous: Mapping[str, tuple[ParamBindingInfo, ...]]
    ambiguous_options: Mapping[str, tuple[ParamBindingInfo, ...]]
    shadowed_bare: frozenset[str]


_Claimant = ParamBindingInfo | None
_Claims = dict[str, tuple[_Claimant, ...]]


def _append_claim(claims: dict[str, list[_Claimant]], spelling: str, claimant: _Claimant) -> None:
    claims.setdefault(spelling, []).append(claimant)


def _freeze_claims(claims: dict[str, list[_Claimant]]) -> _Claims:
    return {spelling: tuple(claimants) for spelling, claimants in claims.items()}


def _param_spellings(param: ParamBindingInfo) -> tuple[str, ...]:
    """Return *param*'s logical long-option names, shortest route first."""
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
    return tuple(unique_spellings)


def _option_spellings(param: ParamBindingInfo, spellings: Iterable[str]) -> tuple[str, ...]:
    """Return every CLI flag projected from *spellings*, followed by its short flag."""
    flags: list[str] = []
    for spelling in spellings:
        projected = project_option(spelling, param.type)
        flags.extend(projected.flags)
        flags.extend(projected.negative_flags)
    if param.cli.short is not None:
        flags.append(f"-{param.cli.short}")
    return tuple(flags)


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
        for flag in (*projected.flags, *projected.negative_flags):
            _append_claim(claims, flag, None)
        if parameter.cli.short is not None:
            _append_claim(claims, f"-{parameter.cli.short}", None)
    return _freeze_claims(claims)


def _param_long_claims(params: Iterable[ParamBindingInfo]) -> _Claims:
    claims: dict[str, list[_Claimant]] = {}
    for param in params:
        for spelling in _param_spellings(param):
            _append_claim(claims, spelling, param)
    return _freeze_claims(claims)


def _param_option_claims(params: Iterable[ParamBindingInfo]) -> _Claims:
    claims: dict[str, list[_Claimant]] = {}
    for param in params:
        for spelling in _option_spellings(param, _param_spellings(param)):
            _append_claim(claims, spelling, param)
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
    all_params = tuple(params)
    own_params = tuple(param for param in all_params if param.module == program.module)
    imported_params = tuple(param for param in all_params if param.module != program.module)

    host_long = _host_long_claims(reserved_flags)
    signature_long = _signature_long_claims(program)
    own_long = _param_long_claims(own_params)
    imported_long = _param_long_claims(imported_params)
    long_winners, ambiguous = _resolve_claims((host_long, signature_long, own_long, imported_long))

    host_options = _host_option_claims(reserved_flags)
    signature_options = _signature_option_claims(program)
    own_options = _param_option_claims(own_params)
    imported_options = _param_option_claims(imported_params)
    option_winners, ambiguous_options = _resolve_claims(
        (host_options, signature_options, own_options, imported_options)
    )

    entries = tuple(
        _entry(param, long_winners, option_winners) for param in (*own_params, *imported_params)
    )
    shadowed_bare = frozenset(
        param.cli.name
        for param in all_params
        if param.cli.name in host_long or param.cli.name in signature_long
    )
    return ParamSurface(
        entries=entries,
        ambiguous=MappingProxyType(ambiguous),
        ambiguous_options=MappingProxyType(ambiguous_options),
        shadowed_bare=shadowed_bare,
    )


def _entry(
    param: ParamBindingInfo,
    long_winners: Mapping[str, ParamBindingInfo | None],
    option_winners: Mapping[str, ParamBindingInfo | None],
) -> ParamSurfaceEntry:
    spellings = tuple(
        spelling for spelling in _param_spellings(param) if long_winners.get(spelling) == param
    )
    option_spellings = tuple(
        flag for flag in _option_spellings(param, spellings) if option_winners.get(flag) == param
    )
    return ParamSurfaceEntry(
        param=param,
        spellings=spellings,
        option_spellings=option_spellings,
        section=param.module.display(),
        hidden=param.cli.hidden,
    )
