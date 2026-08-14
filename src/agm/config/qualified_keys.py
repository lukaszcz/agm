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
            if any(_table_path_replaced(layer, path) for path in _table_paths_for(key)):
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
        for path in _table_paths_for(key):
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
                f"config key {_display_table_path(path)}.{leaf} matches multiple routes: {names}"
            )

    resolved: dict[QualifiedConfigKey, object] = {}
    for key, values in values_by_key.items():
        if len(values) > 1:
            spellings = ", ".join(_display_table_path(path) for path, _ in values)
            raise QualifiedConfigLookupError(
                f"config key {key.display_name()} is set by conflicting tables: {spellings}"
            )
        resolved[key] = values[0][1]
    return resolved


def _table_paths_for(key: QualifiedConfigKey) -> tuple[tuple[str, ...], ...]:
    suffix_paths = [
        (*key.module_segments[-depth:], *key.scope_path)
        for depth in range(1, len(key.module_segments) + 1)
    ]
    anchor_path = ("/".join(key.module_segments), *key.scope_path)
    paths: list[tuple[str, ...]] = []
    seen_paths: set[tuple[str, ...]] = set()
    for path in (*suffix_paths, anchor_path):
        if path[0] not in RESERVED_CONFIG_SECTION_NAMES and path not in seen_paths:
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


def _display_table_path(path: tuple[str, ...]) -> str:
    first, *rest = path
    if "/" in first:
        return ".".join((f'["{first}"]', *rest))
    return ".".join(path)
