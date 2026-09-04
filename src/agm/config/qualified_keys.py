"""Resolve AgL configuration values addressed by qualified module paths."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from agm.command_catalog import RESERVED_COMMAND_NAMES
from agm.config.general import GeneralConfig
from agm.core.toml import TomlDict

# These tables describe AGM's configuration schema rather than AgL modules.
# ``params`` is retained here to prevent the removed legacy ``[params.*]``
# namespace from being interpreted as an AgL module route.
RESERVED_CONFIG_SECTION_NAMES = RESERVED_COMMAND_NAMES | frozenset(
    {"deps", "modules", "packages", "params"}
)
_MISSING = object()


class QualifiedConfigLookupError(ValueError):
    """Raised when a qualified configuration table cannot be resolved uniquely."""


@dataclass(frozen=True)
class QualifiedConfigKey:
    """The module-qualified address of one AgL configuration value.

    ``module_segments`` is the declaring module's slash path, while
    ``scope_path`` and ``leaf`` name the declaration within that module.
    """

    module_segments: tuple[str, ...]
    scope_path: tuple[str, ...]
    leaf: str

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
                for path in route_table_paths(key.module_segments, key.scope_path)
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
    values_by_key: dict[QualifiedConfigKey, list[tuple[tuple[str, ...], object]]] = {}

    for key in keys:
        for path in route_table_paths(key.module_segments, key.scope_path):
            table = _table_at(layer, path)
            if table is None or key.leaf not in table:
                continue
            route = (key.module_segments, key.scope_path)
            matches_by_value.setdefault((path, key.leaf), {}).setdefault(route, []).append(key)
            values_by_key.setdefault(key, []).append((path, table[key.leaf]))

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
            spellings = ", ".join(display_table_path(path) for path, _ in values)
            raise QualifiedConfigLookupError(
                f"config key {key.display_name()} is set by conflicting tables: {spellings}"
            )
        resolved[key] = values[0][1]
    return resolved


def configured_leaf_names(
    config: GeneralConfig,
    module_segments: tuple[str, ...],
    scope_path: tuple[str, ...] = (),
) -> frozenset[str]:
    """Return the leaf key names set in the tables that address one route.

    The tables are exactly the ones :func:`resolve_qualified_values` reads for a
    key on this route, across every layer, so a name is reported only when a
    lookup on that route could observe it. A nested table addresses a route of
    its own — a scope region or a program scope — and is never a leaf here.
    """
    paths = route_table_paths(module_segments, scope_path)
    names: set[str] = set()
    for layer in config.layers:
        for path in paths:
            table = _table_at(layer, path)
            if table is None:
                continue
            names.update(name for name, value in table.items() if not isinstance(value, dict))
    return frozenset(names)


def route_table_paths(
    module_segments: tuple[str, ...], scope_path: tuple[str, ...] = ()
) -> tuple[tuple[str, ...], ...]:
    """Return the config table paths that address one route, in read order.

    Every module-suffix spelling, shortest first, then the exact quoted module
    anchor. AGM's top-level configuration sections are excluded, except that a
    single-segment loose-file module with a reserved stem may address one of
    its nested declaration tables. This is the single routing rule shared by
    value resolution and leaf enumeration, so a caller can tell which routes a
    given table serves.
    """
    suffix_paths = [
        (*module_segments[-depth:], *scope_path) for depth in range(1, len(module_segments) + 1)
    ]
    anchor_path = ("/".join(module_segments), *scope_path)
    paths: list[tuple[str, ...]] = []
    seen_paths: set[tuple[str, ...]] = set()
    for path in (*suffix_paths, anchor_path):
        is_reserved_root = path[0] in RESERVED_CONFIG_SECTION_NAMES
        is_nested_loose_file_route = len(module_segments) == 1 and bool(scope_path)
        if (not is_reserved_root or is_nested_loose_file_route) and path not in seen_paths:
            seen_paths.add(path)
            paths.append(path)
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
