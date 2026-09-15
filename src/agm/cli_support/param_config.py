"""Configuration routes shared by host-configurable module parameters."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import NamedTuple

from agm.agl.ir.static_keys import StaticBindingKey
from agm.agl.modules.ids import ModuleId
from agm.agl.runtime.types import ParamBindingInfo, ProgramDeclInfo
from agm.agl.zones import ParamZone
from agm.cli_support.param_surface import ParamSurface, ParamSurfaceEntry
from agm.cli_support.program_options import native_raw_value, project_option
from agm.config.engine_keys import ENGINE_KEY_NAMES
from agm.config.general import GeneralConfig
from agm.config.qualified_keys import (
    QualifiedConfigKey,
    QualifiedConfigLookupError,
    configured_leaf_tables,
    display_table_path,
    resolve_qualified_values,
)

__all__ = ["RouteReport", "resolve_module_param_values", "resolve_param_values"]


class RouteReport(NamedTuple):
    """One config table route and the leaves it may supply."""

    module_segments: tuple[str, ...]
    scope_path: tuple[str, ...]
    command_paths: tuple[tuple[str, ...], ...]
    declared_leaves: frozenset[str]
    positional_only: frozenset[str]


def _report_undeclared_config_keys(config: GeneralConfig, routes: Iterable[RouteReport]) -> None:
    """Warn for configured route leaves that no host input consumes.

    Each route carries its own declaration set because one config table can
    address a module binding while another addresses the selected program.
    Keeping the existing warning wording preserves the ``agm exec`` contract
    while allowing module routes to use the same reporting primitive.
    """
    for route in routes:
        leaf_tables = configured_leaf_tables(
            config,
            route.module_segments,
            route.scope_path,
            route.command_paths,
        )
        for leaf in sorted(leaf_tables):
            table_name = display_table_path(leaf_tables[leaf])
            if leaf in route.positional_only:
                print(
                    f"warning: config key '{leaf}' in the '{table_name}' configuration table "
                    "names a positional-only program argument, which can only be supplied "
                    "positionally, and will be ignored",
                    file=sys.stderr,
                )
                continue
            if leaf in route.declared_leaves:
                continue
            print(
                f"warning: config key '{leaf}' in the '{table_name}' "
                "configuration table is not a declared program argument and will be ignored",
                file=sys.stderr,
            )


def resolve_param_values(
    config: GeneralConfig,
    program: ProgramDeclInfo,
    params: Mapping[StaticBindingKey, object],
    *,
    entry_segments: tuple[str, ...],
    command_paths: tuple[tuple[str, ...], ...],
    surface: ParamSurface,
) -> tuple[dict[StaticBindingKey, object], list[RouteReport]]:
    """Resolve module-parameter config values beneath parsed CLI/environment values.

    Module routes are addressed by each binding's declaration module and
    scope. A selected program's qualified table can override only a parameter
    whose bare spelling reaches it on *surface*. The two routes are resolved
    independently, so a program route wins even when its value comes from a
    less-specific config layer than the module route.
    """
    values = dict(params)
    supplied = frozenset(params)
    entries = surface.entries
    module_params = tuple(entry.param for entry in entries)
    module_routes = _module_routes(module_params)
    program_routes = _program_routes(program, entry_segments, command_paths, entries)
    _reject_configured_cross_route_ambiguities(config, module_routes, program_routes)
    module_values = resolve_module_param_values(config, module_params)
    for key, value in module_values.items():
        if key not in supplied:
            values[key] = value

    _reject_configured_ambiguous_program_leaves(
        config, program, entry_segments, command_paths, surface
    )
    program_values = resolve_qualified_values(config, tuple(key for _entry, key in program_routes))
    _merge_route_values(values, supplied, program_routes, program_values)
    return values, _route_reports(program, entry_segments, command_paths, entries, program_routes)


def _reject_configured_cross_route_ambiguities(
    config: GeneralConfig,
    module_routes: Sequence[tuple[ParamBindingInfo, QualifiedConfigKey]],
    program_routes: Sequence[tuple[ParamSurfaceEntry, QualifiedConfigKey]],
) -> None:
    """Reject a configured table leaf that resolves to distinct route kinds."""
    for module_param, module_key in module_routes:
        module_leaves = configured_leaf_tables(
            config, module_key.module_segments, module_key.scope_path, module_key.command_paths
        )
        module_table = module_leaves.get(module_key.leaf)
        if module_table is None:
            continue
        for entry, program_key in program_routes:
            if module_param.key == entry.param.key or module_key.leaf != program_key.leaf:
                continue
            program_leaves = configured_leaf_tables(
                config,
                program_key.module_segments,
                program_key.scope_path,
                program_key.command_paths,
            )
            if program_leaves.get(program_key.leaf) != module_table:
                continue
            table_name = display_table_path(module_table)
            raise QualifiedConfigLookupError(
                f"config key {table_name}.{module_key.leaf} matches multiple parameters: "
                f"{module_param.declaration_path}, {entry.param.declaration_path}"
            )


def resolve_module_param_values(
    config: GeneralConfig, params: Sequence[ParamBindingInfo]
) -> dict[StaticBindingKey, object]:
    """Resolve parameter values from their declaration-module routes only.

    This is the config surface used by incremental hosts, which initialize
    modules without selecting a program and therefore have no program route.
    """
    module_routes = _module_routes(params)
    _reject_configured_ambiguous_module_leaves(config, params)
    configured = resolve_qualified_values(config, tuple(key for _param, key in module_routes))
    values: dict[StaticBindingKey, object] = {}
    for param, key in module_routes:
        if key not in configured:
            continue
        projected = project_option(param.cli.name, param.type)
        values[param.key] = native_raw_value(projected, configured[key])
    return values


def _module_routes(
    params: Sequence[ParamBindingInfo],
) -> tuple[tuple[ParamBindingInfo, QualifiedConfigKey], ...]:
    """Return declaration-module routes for every non-anonymous binding."""
    return tuple(
        (
            param,
            QualifiedConfigKey(
                param.module.segments,
                param.scope_path,
                param.cli.name,
            ),
        )
        for param in params
        if not param.module.is_entry
    )


def _reject_configured_ambiguous_module_leaves(
    config: GeneralConfig, params: Sequence[ParamBindingInfo]
) -> None:
    """Reject a configured leaf claimed by multiple bindings on one module route."""
    candidates_by_route: dict[
        tuple[tuple[str, ...], tuple[str, ...], str], list[ParamBindingInfo]
    ] = {}
    for param in params:
        if param.module.is_entry:
            continue
        route = (param.module.segments, param.scope_path, param.cli.name)
        candidates_by_route.setdefault(route, []).append(param)
    for (module_segments, scope_path, leaf), candidates in candidates_by_route.items():
        if len(candidates) < 2:
            continue
        leaf_tables = configured_leaf_tables(config, module_segments, scope_path)
        if leaf not in leaf_tables:
            continue
        names = ", ".join(candidate.declaration_path for candidate in candidates)
        table_name = display_table_path(leaf_tables[leaf])
        raise QualifiedConfigLookupError(
            f"config key {table_name}.{leaf} matches multiple parameters: {names}"
        )


def _program_routes(
    program: ProgramDeclInfo,
    entry_segments: tuple[str, ...],
    command_paths: tuple[tuple[str, ...], ...],
    entries: Sequence[ParamSurfaceEntry],
) -> tuple[tuple[ParamSurfaceEntry, QualifiedConfigKey], ...]:
    """Return selected-program routes for bare module-parameter spellings."""
    if not entry_segments:
        return ()
    program_path = (*program.scope_path, program.name)
    return tuple(
        (
            entry,
            QualifiedConfigKey(entry_segments, program_path, entry.param.cli.name, command_paths),
        )
        for entry in entries
        if entry.param.cli.name in entry.spellings and entry.param.cli.name not in ENGINE_KEY_NAMES
    )


def _merge_route_values(
    values: dict[StaticBindingKey, object],
    supplied: frozenset[StaticBindingKey],
    routes: Iterable[tuple[ParamSurfaceEntry, QualifiedConfigKey]],
    configured: Mapping[QualifiedConfigKey, object],
) -> None:
    """Project configured native values without replacing CLI/environment values."""
    for entry, key in routes:
        if entry.param.key in supplied or key not in configured:
            continue
        projected = project_option(entry.param.cli.name, entry.param.type)
        values[entry.param.key] = native_raw_value(projected, configured[key])


def _reject_configured_ambiguous_program_leaves(
    config: GeneralConfig,
    program: ProgramDeclInfo,
    entry_segments: tuple[str, ...],
    command_paths: tuple[tuple[str, ...], ...],
    surface: ParamSurface,
) -> None:
    """Reject a configured program-table leaf claimed by peer module parameters."""
    if not entry_segments:
        return
    program_path = (*program.scope_path, program.name)
    configured = configured_leaf_tables(config, entry_segments, program_path, command_paths)
    for spelling, candidates in surface.ambiguous.items():
        if spelling not in configured or any(
            candidate.cli.name != spelling for candidate in candidates
        ):
            continue
        names = ", ".join(candidate.declaration_path for candidate in candidates)
        table_name = display_table_path(configured[spelling])
        raise QualifiedConfigLookupError(
            f"config key {table_name}.{spelling} matches multiple parameters: {names}"
        )


def _route_reports(
    program: ProgramDeclInfo,
    entry_segments: tuple[str, ...],
    command_paths: tuple[tuple[str, ...], ...],
    entries: Sequence[ParamSurfaceEntry],
    program_routes: Sequence[tuple[ParamSurfaceEntry, QualifiedConfigKey]],
) -> list[RouteReport]:
    """Describe module and selected-program config routes for warning reporting."""
    by_module_scope: dict[tuple[ModuleId, tuple[str, ...]], list[ParamBindingInfo]] = {}
    for entry in entries:
        if not entry.param.module.is_entry:
            by_module_scope.setdefault((entry.param.module, entry.param.scope_path), []).append(
                entry.param
            )

    reports: list[RouteReport] = []
    ordered_modules = _report_modules(program, entries)
    for module in ordered_modules:
        if module.is_entry:
            continue
        root_params = by_module_scope.get((module, ()), [])
        reports.append(
            RouteReport(
                module.segments,
                (),
                (),
                frozenset(param.cli.name for param in root_params),
                frozenset(),
            )
        )
        scope_paths = tuple(
            scope_path
            for candidate, scope_path in by_module_scope
            if candidate == module and scope_path
        )
        for scope_path in scope_paths:
            declared = frozenset(param.cli.name for param in by_module_scope[(module, scope_path)])
            reports.append(RouteReport(module.segments, scope_path, (), declared, frozenset()))

    if entry_segments:
        declared = frozenset(
            {
                *(parameter.cli.name for parameter in program.parameters),
                *(entry.param.cli.name for entry, _key in program_routes),
                *ENGINE_KEY_NAMES,
            }
        )
        positional_only = frozenset(
            parameter.cli.name
            for parameter in program.parameters
            if parameter.kind is ParamZone.POSITIONAL_ONLY
        )
        reports.append(
            RouteReport(
                entry_segments,
                (*program.scope_path, program.name),
                command_paths,
                declared,
                positional_only,
            )
        )
    return reports


def _report_modules(
    program: ProgramDeclInfo, entries: Sequence[ParamSurfaceEntry]
) -> tuple[ModuleId, ...]:
    """Return closure modules first, retaining any surface-only module afterward."""
    modules: list[ModuleId] = list(program.closure)
    for entry in entries:
        if entry.param.module not in modules:
            modules.append(entry.param.module)
    return tuple(modules)
