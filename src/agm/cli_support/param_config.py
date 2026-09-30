"""Configuration routes shared by host-configurable module parameters."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
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
    configured_leaf_table_candidates,
    configured_leaf_tables,
    display_table_path,
    manifest_leaf_tables,
    resolve_manifest_values,
    resolve_qualified_values,
)
from agm.core.toml import TomlDict

__all__ = [
    "ParamValueTiers",
    "resolve_module_param_values",
    "resolve_param_values",
]


#: A config table route: declaring module, scope, and registered command paths.
_RouteKey = tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, ...], ...]]


@dataclass(frozen=True)
class ParamValueTiers:
    """Supplied/program-route values (``upper``) above module-route values (``lower``).

    ``lower`` holds only keys absent from ``upper``, and within it a
    package manifest ``[config]`` value (when the selected program is
    package-owned) already outranks the plain module-route value for the
    same key. The full chain also ranks a selected program's own ``@config``
    entries between the two — ``PipelineDriver.preflight_arguments`` merges
    all three; there is no flattened two-tier view here.
    """

    upper: Mapping[StaticBindingKey, object]
    lower: Mapping[StaticBindingKey, object]


class _RouteReport(NamedTuple):
    """One config table route and the leaves it may supply."""

    module_segments: tuple[str, ...]
    scope_path: tuple[str, ...]
    command_paths: tuple[tuple[str, ...], ...]
    declared_leaves: frozenset[str]
    positional_only: frozenset[str]


def _tables_with_registered_descendants(
    command_paths: tuple[tuple[str, ...], ...],
    package_command_paths: frozenset[tuple[str, ...]],
) -> frozenset[tuple[str, ...]]:
    """Return which of *command_paths* is a proper prefix of another registered command.

    Such a table's own leaves may be inherited defaults meant for that
    descendant command (see the group-table inheritance in
    :func:`agm.config.qualified_keys.resolve_qualified_values`), so a warning
    on this command must leave them alone — a sibling command may be the one
    that actually consumes them.
    """
    return frozenset(
        path
        for path in command_paths
        if any(
            len(other) > len(path) and other[: len(path)] == path for other in package_command_paths
        )
    )


def _report_undeclared_config_keys(
    config: GeneralConfig,
    routes: Iterable[_RouteReport],
    package_command_paths: frozenset[tuple[str, ...]],
) -> None:
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
        exempt_tables = _tables_with_registered_descendants(
            route.command_paths, package_command_paths
        )
        for leaf in sorted(leaf_tables):
            table_path = leaf_tables[leaf]
            if table_path in exempt_tables:
                continue
            table_name = display_table_path(table_path)
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


def _report_unused_manifest_leaves(
    package_config: TomlDict,
    command_paths: tuple[tuple[str, ...], ...],
    consumed: frozenset[str],
    package_command_paths: frozenset[tuple[str, ...]],
) -> None:
    """Warn for manifest leaves in the selected command's own tables that nothing consumes.

    Engine settings are read by the host and *consumed* holds the program-route
    spellings of the reachable module parameters. Inherited group and root
    tables are skipped, since a sibling command may be the one that consumes
    them, as are tables a registered descendant command inherits from.
    """
    exempt_tables = _tables_with_registered_descendants(command_paths, package_command_paths)
    own_tables = {("config", *path) for path in command_paths} - {
        ("config", *path) for path in exempt_tables
    }
    leaf_tables = manifest_leaf_tables(package_config, command_paths)
    for leaf in sorted(leaf_tables):
        table_path = leaf_tables[leaf]
        if table_path not in own_tables or leaf in ENGINE_KEY_NAMES or leaf in consumed:
            continue
        print(
            f"warning: config key '{leaf}' in the '{display_table_path(table_path)}' "
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
    package_command_paths: frozenset[tuple[str, ...]] = frozenset(),
    package_config: TomlDict | None = None,
) -> ParamValueTiers:
    """Resolve module-parameter config values beneath parsed CLI/environment values.

    Module routes are addressed by each binding's declaration module and
    scope. A selected program's qualified table overrides a parameter through
    any spelling it wins on *surface*: its bare name, or a qualified spelling
    when a nearer declaration claims that name. The two routes are resolved
    independently, so a program route wins even when its value comes from a
    less-specific config layer than the module route. The supplied and
    program-route values form :attr:`ParamValueTiers.upper`; the module-route
    values not already covered by that tier form :attr:`ParamValueTiers.lower`.

    *package_config* is the owning package's manifest ``[config]`` table
    (``None``/empty when the program is not package-owned). Its values
    address the same program-route spellings and are merged into ``lower``,
    above the plain module-route value, for every key not already resolved
    in ``upper`` — never overriding an ``upper`` entry.

    Configured leaves on those routes that no host input consumes are reported
    as warnings here, where the routes are known. *package_command_paths* is
    every command path the owning package registers (aliases expanded), used
    to exempt a command's own table from that warning when one of its leaves
    is really an inherited default for a registered descendant command.
    """
    supplied = frozenset(params)
    entries = surface.entries
    module_params = tuple(entry.param for entry in entries)
    module_routes = _module_routes(module_params)
    program_routes = _program_routes(program, entry_segments, command_paths, entries)
    _reject_configured_cross_route_ambiguities(config, module_routes, program_routes)
    module_values = resolve_module_param_values(config, module_params)

    _reject_configured_ambiguous_program_leaves(
        config, program, entry_segments, command_paths, surface, package_config
    )
    program_values = resolve_qualified_values(config, tuple(key for _entry, key in program_routes))
    upper = dict(params)
    _merge_route_values(upper, supplied, program_routes, program_values)
    lower = {key: value for key, value in module_values.items() if key not in upper}
    if package_config:
        manifest_values = resolve_manifest_values(
            package_config, tuple(key for _entry, key in program_routes)
        )
        _merge_route_values(lower, frozenset(upper), program_routes, manifest_values)
        _report_unused_manifest_leaves(
            package_config,
            command_paths,
            frozenset(
                spelling for _entry, key in program_routes for spelling in key.leaf_spellings()
            ),
            package_command_paths,
        )

    _report_undeclared_config_keys(
        config,
        _route_reports(program, entry_segments, command_paths, entries, program_routes),
        package_command_paths,
    )
    return ParamValueTiers(upper=upper, lower=lower)


def _reject_configured_cross_route_ambiguities(
    config: GeneralConfig,
    module_routes: Sequence[tuple[ParamBindingInfo, QualifiedConfigKey]],
    program_routes: Sequence[tuple[ParamSurfaceEntry, QualifiedConfigKey]],
) -> None:
    """Reject a configured table leaf that resolves to distinct route kinds.

    The program side includes tables inherited from a command path's group
    prefixes, since a leaf set there is just as ambiguous against a module
    route as one set on the exact route.
    """
    if not program_routes:
        return
    # Every program route addresses the same table route, so its leaves are
    # read once; module routes repeat per declaration module and scope.
    _entry, shared_key = program_routes[0]
    program_leaves = configured_leaf_table_candidates(
        config,
        shared_key.module_segments,
        shared_key.scope_path,
        shared_key.command_paths,
    )
    module_leaves_by_route: dict[_RouteKey, dict[str, tuple[str, ...]]] = {}
    for module_param, module_key in module_routes:
        route: _RouteKey = (
            module_key.module_segments,
            module_key.scope_path,
            module_key.command_paths,
        )
        if route not in module_leaves_by_route:
            module_leaves_by_route[route] = configured_leaf_tables(
                config, module_key.module_segments, module_key.scope_path, module_key.command_paths
            )
        module_leaves = module_leaves_by_route[route]
        module_table = module_leaves.get(module_key.leaf)
        if module_table is None:
            continue
        peers = tuple(
            entry.param
            for entry, program_key in program_routes
            if module_param.key != entry.param.key
            and module_key.leaf == program_key.leaf
            and module_table in program_leaves.get(program_key.leaf, frozenset())
        )
        if peers:
            _reject_configured_ambiguity(module_leaves, {module_key.leaf: (module_param, *peers)})


def _reject_configured_ambiguity(
    leaf_tables: Mapping[str, tuple[str, ...]],
    candidates: Mapping[str, Sequence[ParamBindingInfo]],
) -> None:
    """Reject the first *candidates* leaf that one of *leaf_tables*' tables sets."""
    for leaf, claimants in candidates.items():
        table = leaf_tables.get(leaf)
        if table is None:
            continue
        names = ", ".join(claimant.declaration_path for claimant in claimants)
        raise QualifiedConfigLookupError(
            f"config key {display_table_path(table)}.{leaf} matches multiple parameters: {names}"
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
        _reject_configured_ambiguity(
            configured_leaf_tables(config, module_segments, scope_path), {leaf: candidates}
        )


def _program_routes(
    program: ProgramDeclInfo,
    entry_segments: tuple[str, ...],
    command_paths: tuple[tuple[str, ...], ...],
    entries: Sequence[ParamSurfaceEntry],
) -> tuple[tuple[ParamSurfaceEntry, QualifiedConfigKey], ...]:
    """Return selected-program routes for resolving module-parameter spellings.

    Every spelling a parameter wins on the CLI surface addresses it in the
    selected program's table too, so a parameter whose bare name is claimed by
    a nearer declaration — the host's engine keys included — stays configurable
    per program through a qualified spelling. :class:`ParamSurface` is the sole
    authority on which spellings those are.
    """
    if not entry_segments:
        return ()
    program_path = (*program.scope_path, program.name)
    routes: list[tuple[ParamSurfaceEntry, QualifiedConfigKey]] = []
    for entry in entries:
        if not entry.spellings:
            continue
        leaf, *aliases = entry.spellings
        routes.append(
            (
                entry,
                QualifiedConfigKey(
                    entry_segments, program_path, leaf, command_paths, tuple(aliases)
                ),
            )
        )
    return tuple(routes)


def _merge_route_values(
    values: dict[StaticBindingKey, object],
    supplied: frozenset[StaticBindingKey],
    routes: Iterable[tuple[ParamSurfaceEntry, QualifiedConfigKey]],
    configured: Mapping[QualifiedConfigKey, object],
) -> None:
    """Project configured native values into *values*, skipping keys already in *supplied*.

    Shared by the program-route merge into ``upper`` (*supplied* = CLI/
    environment keys) and the manifest merge into ``lower`` (*supplied* =
    every key already resolved in ``upper``, so a manifest value never
    overrides one): both are "does a higher tier already own this key".
    """
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
    package_config: TomlDict | None = None,
) -> None:
    """Reject a configured program-table leaf claimed by peer module parameters.

    Every spelling a program table may use is checked, bare or qualified: a
    spelling several parameters claim resolves to none of them, so *using* it
    names the candidates rather than silently picking one. *package_config*
    extends the same check over the owning package's manifest command, group,
    and root tables. Config group tables are covered too, since an inherited
    leaf is just as ambiguous as one set on the exact route.
    """
    if not entry_segments:
        return
    program_path = (*program.scope_path, program.name)
    candidates = configured_leaf_table_candidates(
        config, entry_segments, program_path, command_paths
    )
    configured = {leaf: min(tables) for leaf, tables in candidates.items()}
    if package_config:
        configured = {**manifest_leaf_tables(package_config, command_paths), **configured}
    _reject_configured_ambiguity(configured, surface.ambiguous)


def _route_reports(
    program: ProgramDeclInfo,
    entry_segments: tuple[str, ...],
    command_paths: tuple[tuple[str, ...], ...],
    entries: Sequence[ParamSurfaceEntry],
    program_routes: Sequence[tuple[ParamSurfaceEntry, QualifiedConfigKey]],
) -> list[_RouteReport]:
    """Describe module and selected-program config routes for warning reporting."""
    by_module_scope: dict[tuple[ModuleId, tuple[str, ...]], list[ParamBindingInfo]] = {}
    for entry in entries:
        if not entry.param.module.is_entry:
            by_module_scope.setdefault((entry.param.module, entry.param.scope_path), []).append(
                entry.param
            )

    reports: list[_RouteReport] = []
    ordered_modules = _report_modules(program, entries)
    for module in ordered_modules:
        if module.is_entry:
            continue
        root_params = by_module_scope.get((module, ()), [])
        reports.append(
            _RouteReport(
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
            reports.append(_RouteReport(module.segments, scope_path, (), declared, frozenset()))

    if entry_segments:
        declared = frozenset(
            {
                *(parameter.cli.name for parameter in program.parameters),
                *(spelling for _entry, key in program_routes for spelling in key.leaf_spellings()),
                *ENGINE_KEY_NAMES,
            }
        )
        positional_only = frozenset(
            parameter.cli.name
            for parameter in program.parameters
            if parameter.kind is ParamZone.POSITIONAL_ONLY
        )
        reports.append(
            _RouteReport(
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
