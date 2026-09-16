"""Resolve AgL configuration values addressed by qualified module paths."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from agm.command_catalog import RESERVED_COMMAND_NAMES
from agm.config.general import GeneralConfig
from agm.core.toml import TomlDict

# These tables describe AGM's configuration schema rather than AgL modules, and
# key their own nested tables by user-chosen names — a dependency, a module
# root, a package pin. ``params`` is retained here to prevent the removed legacy
# ``[params.*]`` namespace from being interpreted as an AgL module route.
SCHEMA_CONFIG_SECTION_NAMES = frozenset({"deps", "modules", "packages", "params"})
RESERVED_CONFIG_SECTION_NAMES = RESERVED_COMMAND_NAMES | SCHEMA_CONFIG_SECTION_NAMES
_MISSING = object()


class QualifiedConfigLookupError(ValueError):
    """Raised when a qualified configuration table cannot be resolved uniquely."""


@dataclass(frozen=True)
class QualifiedConfigKey:
    """The module-qualified address of one AgL configuration value.

    ``module_segments`` is the declaring module's slash path, while
    ``scope_path`` and ``leaf`` name the declaration within that module.
    ``command_paths`` carries the CLI paths a package registers for this
    declaration, each of which addresses it as well as its module route does.
    ``leaf_aliases`` holds further leaf spellings that address the same
    declaration within those tables, so one key covers a module parameter's
    bare and qualified program-table spellings under the ordinary layering
    rules: several spellings in one layer conflict, a later layer wins.
    """

    module_segments: tuple[str, ...]
    scope_path: tuple[str, ...]
    leaf: str
    command_paths: tuple[tuple[str, ...], ...] = ()
    leaf_aliases: tuple[str, ...] = ()

    def leaf_spellings(self) -> tuple[str, ...]:
        """Return every leaf spelling that addresses this declaration."""

        return (self.leaf, *self.leaf_aliases)

    def display_name(self) -> str:
        """Return the source-style spelling used in lookup diagnostics."""

        module_path = "/".join(self.module_segments)
        return "::".join((module_path, *self.scope_path, self.leaf))


def resolve_qualified_values(
    config: GeneralConfig, keys: Iterable[QualifiedConfigKey]
) -> dict[QualifiedConfigKey, object]:
    """Resolve *keys* from the normalized layers of *config*.

    Values in later layers replace values from earlier ones, including when the
    layers use different valid suffix spellings. Each individual layer still
    rejects conflicting spellings. Requiring :class:`GeneralConfig` prevents a
    caller from accidentally resolving the merged view after layer provenance
    has been lost.
    """

    layers = config.layers
    # ``dict.fromkeys`` would be shorter but is untyped under the repo's
    # ``disallow_any_expr`` setting.
    seen_keys: set[QualifiedConfigKey] = set()
    unique_keys_list: list[QualifiedConfigKey] = []
    for key in keys:
        if key not in seen_keys:
            seen_keys.add(key)
            unique_keys_list.append(key)
    unique_keys = tuple(unique_keys_list)

    resolved: dict[QualifiedConfigKey, object] = {}
    for layer in layers:
        for key in unique_keys:
            if any(
                _table_path_replaced(layer, path)
                for path in route_table_paths(
                    key.module_segments, key.scope_path, key.command_paths
                )
            ):
                resolved.pop(key, None)
        resolved.update(_resolve_layer(layer, unique_keys))
    return resolved


def _resolve_layer(
    layer: TomlDict, keys: tuple[QualifiedConfigKey, ...]
) -> dict[QualifiedConfigKey, object]:
    matches_by_value: dict[
        tuple[tuple[str, ...], str],
        dict[tuple[tuple[str, ...], tuple[str, ...]], list[QualifiedConfigKey]],
    ] = {}
    values_by_key: dict[QualifiedConfigKey, list[tuple[tuple[str, ...], str, object]]] = {}

    for key in keys:
        for path in route_table_paths(key.module_segments, key.scope_path, key.command_paths):
            table = _table_at(layer, path)
            if table is None:
                continue
            for leaf in key.leaf_spellings():
                if leaf not in table:
                    continue
                route = (key.module_segments, key.scope_path)
                matches_by_value.setdefault((path, leaf), {}).setdefault(route, []).append(key)
                values_by_key.setdefault(key, []).append((path, leaf, table[leaf]))

    for (path, leaf), candidates_by_route in matches_by_value.items():
        if len(candidates_by_route) > 1:
            names = ", ".join(
                key.display_name()
                for candidates in candidates_by_route.values()
                for key in candidates
            )
            raise QualifiedConfigLookupError(
                f"config key {display_table_path(path)}.{leaf} matches multiple routes: {names}"
            )

    resolved: dict[QualifiedConfigKey, object] = {}
    for key, values in values_by_key.items():
        if len(values) > 1:
            spellings = ", ".join(f"{display_table_path(path)}.{leaf}" for path, leaf, _ in values)
            raise QualifiedConfigLookupError(
                f"config key {key.display_name()} is set by conflicting tables: {spellings}"
            )
        resolved[key] = values[0][2]
    return resolved


def configured_leaf_tables(
    config: GeneralConfig,
    module_segments: tuple[str, ...],
    scope_path: tuple[str, ...] = (),
    command_paths: tuple[tuple[str, ...], ...] = (),
) -> dict[str, tuple[str, ...]]:
    """Return each leaf key set on one route, mapped to the table that sets it.

    The tables are exactly the ones :func:`resolve_qualified_values` reads for a
    key on this route, across every layer, so a name is reported only when a
    lookup on that route could observe it. A nested table addresses a route of
    its own — a scope region or a program scope — and is never a leaf here.

    A leaf reported with the table it was actually read from lets a diagnostic
    name the spelling its reader wrote, rather than whichever spelling this
    route happens to consult first.
    """
    paths = route_table_paths(module_segments, scope_path, command_paths)
    tables: dict[str, tuple[str, ...]] = {}
    for layer in config.layers:
        for path in paths:
            table = _table_at(layer, path)
            if table is None:
                continue
            for name, value in table.items():
                if not isinstance(value, dict):
                    tables.setdefault(name, path)
    return tables


def route_table_paths(
    module_segments: tuple[str, ...],
    scope_path: tuple[str, ...] = (),
    command_paths: tuple[tuple[str, ...], ...] = (),
) -> tuple[tuple[str, ...], ...]:
    """Return the config table paths that address one route, in read order.

    Every module-suffix spelling, shortest first, then the exact quoted module
    anchor, then each *command path* the declaration is registered under, whose
    own segments are the whole table path: ``agm dev review`` reads
    ``[dev.review]``. AGM's top-level configuration sections are excluded,
    except that a
    single-segment loose-file module named after a *command* may address one of
    its nested declaration tables: a command section holds its own settings as
    leaf keys, so a table one level below it is free. A section keyed by AGM's
    own schema (:data:`SCHEMA_CONFIG_SECTION_NAMES`) is excluded at every
    depth, because its nested tables already belong to user-chosen names, so a
    loose file named ``packages.agl`` has no config table for its programs at
    all. This is the single routing rule shared by value resolution and leaf
    enumeration, so a caller can tell which routes a given table serves.
    """
    suffix_paths = [
        (*module_segments[-depth:], *scope_path) for depth in range(1, len(module_segments) + 1)
    ]
    anchor_path = ("/".join(module_segments), *scope_path)
    paths: list[tuple[str, ...]] = []
    seen_paths: set[tuple[str, ...]] = set()
    for path in (*suffix_paths, anchor_path):
        is_reserved_root = path[0] in RESERVED_CONFIG_SECTION_NAMES
        is_nested_loose_file_route = (
            len(module_segments) == 1
            and bool(scope_path)
            and path[0] not in SCHEMA_CONFIG_SECTION_NAMES
        )
        if (not is_reserved_root or is_nested_loose_file_route) and path not in seen_paths:
            seen_paths.add(path)
            paths.append(path)
    for command_path in command_paths:
        # A registered command path never starts at a reserved section: package
        # validation rejects such a registration, so a command table cannot
        # collide with AGM's own configuration schema.
        if command_path not in seen_paths:
            seen_paths.add(command_path)
            paths.append(command_path)
    return tuple(paths)


def _table_at(config: TomlDict, path: tuple[str, ...]) -> dict[str, object] | None:
    current: object = config
    for segment in path:
        if not isinstance(current, dict):
            return None
        current = current.get(segment, _MISSING)
        if current is _MISSING:
            return None
    return current if isinstance(current, dict) else None


def _table_path_replaced(config: TomlDict, path: tuple[str, ...]) -> bool:
    """Return whether *config* explicitly makes *path* non-tabular."""
    current: object = config
    for segment in path:
        if not isinstance(current, dict) or segment not in current:
            return False
        current = current[segment]
        if not isinstance(current, dict):
            return True
    return False


def display_table_path(path: tuple[str, ...]) -> str:
    """Render *path* as the dot-joined TOML table spelling a user would write.

    A path whose first segment is a slash-joined module route (the anchor
    spelling from :func:`route_table_paths`) is rendered as a single quoted
    key (``["a/b"].c``) since a bare ``a/b.c`` is not valid TOML; every other
    path is plain dot-joined (``a.b``).
    """
    first, *rest = path
    if "/" in first:
        return ".".join((f'["{first}"]', *rest))
    return ".".join(path)
