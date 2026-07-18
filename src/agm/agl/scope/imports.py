"""Pure import-resolution models for the AgL module system.

This module provides import environments and qualified-reference resolution.
They compute from :class:`~agm.agl.syntax.nodes.ImportDecl` declarations and
pre-built per-module export sets, with no dependency on the resolver or
scope-tree. They depend only on AST node types, :class:`ModuleId`, and
:class:`AglScopeError`.

This isolation lets the combinatorial import-merge logic be tested exhaustively
in unit tests without instantiating a full resolver.

Design decisions implemented
-----------------------------
- All import forms: bare/``as``/``qualified``/``qualified as``/
  ``using``/``using…as``/``hiding``/wildcard (``.*``)/wildcard ``as`` re-rooting.
- ``private`` names are absent from ``exports``; ``build_import_env`` never
  re-admits them (it operates exclusively on the supplied ``exports`` map).
- Unified namespace: any top-level non-private ``FuncDef``, ``RecordDef``,
  ``EnumDef``, or ``TypeAlias`` name may appear in ``using``/``hiding``.

The resolver and typechecker consume :class:`ImportEnv`. The contribution
model represents per-module member selection, bare injection, and suffix or
anchored qualification.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from agm.agl.modules.ids import ModuleId
from agm.agl.scope.symbols import AglScopeError
from agm.agl.syntax.nodes import ImportDecl
from agm.agl.syntax.types import ImportMode

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

#: A qualified name: (owning module, original name in that module).
QName = tuple[ModuleId, str]


# ---------------------------------------------------------------------------
# ImportTarget — the resolved target of one ImportDecl
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SingleTarget:
    """The import resolves to exactly one module."""

    module: ModuleId


@dataclass(frozen=True, slots=True)
class WildcardTarget:
    """The import wildcard expanded to a set of modules."""

    modules: frozenset[ModuleId]


ImportTarget = SingleTarget | WildcardTarget


# ---------------------------------------------------------------------------
# ImportEnv — the output
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ImportEnv:
    """The import environment for one module, derived from its import declarations.

    ``unqualified``
        Maps each exposed name to the set of distinct :data:`QName` targets that
        expose it.  Size > 1 means a deferred clash-on-use (different modules
        both export the name); the resolver diagnoses the clash *at the reference
        site*, not here.  Identical ``(module, original_name)`` pairs are
        idempotent (imported more than once from the same module is fine).

    ``qualified``
        Maps each qualifier handle (a ``tuple[str, …]`` matching what the user
        writes before ``::``; never empty) to a dict of
        ``exposed_name → QName``.  A name is qualified-accessible *only* if it
        appears here.  The imported set S per (decl, module) therefore bounds
        which names appear.

    Empty-segment handles (``()``) are never stored here; ``::name``
    (self-reference to the current module) is the resolver's responsibility.
    """

    unqualified: Mapping[str, frozenset[QName]]
    qualified: Mapping[tuple[str, ...], Mapping[str, QName]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _module_path(mid: ModuleId) -> str:
    """Return the display slash path of a module (for error messages)."""
    return mid.path_str()


def _module_sort_key(module: ModuleId) -> tuple[str, ...]:
    """Return the deterministic order key for a module identity."""
    return module.segments


def _compute_s(
    decl: ImportDecl,
    module: ModuleId,
    module_exports: frozenset[str],  # set of exported names (keys of the exports map)
    *,
    is_wildcard: bool,
) -> frozenset[str]:
    """Compute S(decl, module) — the imported set for one (decl, module) pair.

    For wildcard imports, cross-module validation (every listed name exported by
    ≥1 matched module) must have already been done by the caller via
    :func:`_validate_wildcard_items` before calling this function.

    Raises :class:`~agm.agl.scope.symbols.AglScopeError` for:
    - ``USING`` single import: name not in ``module_exports``.
    - ``HIDING`` single import: name not in ``module_exports``.
    """
    if decl.mode == ImportMode.ALL:
        return module_exports

    if decl.mode == ImportMode.USING:
        result: set[str] = set()
        for item in decl.items:
            if is_wildcard:
                # Wildcard: include only if this module exports the name.
                # Cross-module validation already done by _validate_wildcard_items.
                if item.name in module_exports:
                    result.add(item.name)
            else:
                # Single import: the name MUST be exported.
                if item.name not in module_exports:
                    raise AglScopeError(
                        f"name {item.name!r} is not exported by module {_module_path(module)!r}",
                        span=decl.span,
                    )
                result.add(item.name)
        return frozenset(result)

    # ImportMode.HIDING
    hidden: set[str] = set()
    for item in decl.items:
        if is_wildcard:
            # Defer cross-module validation to _validate_wildcard_items.
            hidden.add(item.name)
        else:
            if item.name not in module_exports:
                raise AglScopeError(
                    f"name {item.name!r} is not exported by module {_module_path(module)!r}",
                    span=decl.span,
                )
            hidden.add(item.name)
    return module_exports - hidden


def _validate_wildcard_items(
    decl: ImportDecl,
    all_exports: Mapping[ModuleId, frozenset[str]],  # name-only sets (keys of export maps)
) -> None:
    """For wildcard ``using``/``hiding``: every listed name must appear in ≥1 matched module.

    Raises :class:`~agm.agl.scope.symbols.AglScopeError` if any listed name is
    not exported by *any* of the matched modules.
    """
    if decl.mode not in (ImportMode.USING, ImportMode.HIDING):
        return
    union_exports: frozenset[str] = frozenset().union(*all_exports.values())
    for item in decl.items:
        if item.name not in union_exports:
            raise AglScopeError(
                f"name {item.name!r} is not exported by any module matched by the wildcard import",
                span=decl.span,
            )


def _exposed_name(src_name: str, rename_map: dict[str, str]) -> str:
    """Return the exposed name for a source name, applying renames if present.

    ``rename_map`` maps source names to their canonical exposed names.  For
    ``ALL``/``HIDING`` modes (no renames), the map is empty.  For ``USING``
    mode, only renamed items appear in the map; unrenamed items fall through
    to ``src_name``.
    """
    return rename_map.get(src_name, src_name)


def _build_rename_map(decl: ImportDecl) -> dict[str, str]:
    """Build a source-name → exposed-name map for a ``USING`` ImportDecl.

    Returns an empty dict for ``ALL`` and ``HIDING`` modes (no renames).
    Only items that carry an explicit ``rename`` contribute a mapping.
    """
    if decl.mode != ImportMode.USING:
        return {}
    return {item.name: item.rename for item in decl.items if item.rename is not None}


def _qualifier_handle(decl: ImportDecl, module: ModuleId) -> tuple[str, ...]:
    """Compute the qualifier handle for a (decl, module) pair.

    Rules:
    - single, no alias → ``module.segments``
    - single, alias → ``(alias,)``
    - wildcard, no alias → ``module.segments``
    - wildcard, alias → prefix re-rooting: ``(alias,) + module.segments[len(decl.module_path):]``
    """
    if decl.alias is None:
        return module.segments

    # alias present
    if not decl.wildcard:
        return (decl.alias,)

    # wildcard + alias: re-root the matched prefix
    prefix_len = len(decl.module_path)
    tail = module.segments[prefix_len:]
    return (decl.alias,) + tail


# ---------------------------------------------------------------------------
# Alias-root collision detection
# ---------------------------------------------------------------------------


def _check_alias_root_collision(
    decls: tuple[ImportDecl, ...],
    targets: Mapping[int, ImportTarget],
) -> None:
    """Detect alias-root vs non-alias module-path root collision.

    A collision occurs ONLY when at least one entry is an alias.  Specifically,
    a pair (entry_A, entry_B) is a collision when:
    - their ``_qualifier_root`` strings are equal (first segment of the handle), AND
    - at least one of them is an alias (``decl.alias is not None``), AND
    - they denote different module-id sets.

    Two plain (non-alias) dotpath imports sharing a first path segment can NEVER
    collide — their full qualifier handles differ (e.g. ``foo.bar`` vs ``foo.baz``).

    Examples
    --------
    * ``import foo`` + ``import bar.baz as foo`` → collision (foo alias ≠ foo module).
    * ``import foo.bar`` + ``import foo.baz`` → NO collision (siblings, no alias).
    * ``import foo`` + ``import foo.bar`` → NO collision (parent + child, no alias).
    """

    def _target_ids(target: ImportTarget) -> frozenset[ModuleId]:
        if isinstance(target, SingleTarget):
            return frozenset({target.module})
        return target.modules

    # root_segment → list of (target_ids, is_alias, decl) triples sharing that root
    root_entries: dict[str, list[tuple[frozenset[ModuleId], bool, ImportDecl]]] = defaultdict(list)

    for decl in decls:
        target = targets[decl.node_id]
        ids = _target_ids(target)
        is_alias = decl.alias is not None
        root: str = decl.alias if decl.alias is not None else decl.module_path[0]
        root_entries[root].append((ids, is_alias, decl))

    for root, entries in root_entries.items():
        if len(entries) < 2:
            continue
        # Only check pairs where at least one entry is an alias.
        for i, (ids_i, is_alias_i, decl_i) in enumerate(entries):
            for ids_j, is_alias_j, decl_j in entries[i + 1 :]:
                if not (is_alias_i or is_alias_j):
                    # Both are plain dotpath imports — full handles differ; no collision.
                    continue
                if ids_i != ids_j:
                    raise AglScopeError(
                        f"qualifier root {root!r} is ambiguous: used both as a module-path root"
                        f" and as an alias (or two different aliases) that denote different"
                        f" modules",
                        span=decl_j.span,
                    )


# ---------------------------------------------------------------------------
# Alias-to-module binding check (for non-wildcard single alias)
# ---------------------------------------------------------------------------


def _check_duplicate_single_alias(
    decls: tuple[ImportDecl, ...],
    targets: Mapping[int, ImportTarget],
) -> None:
    """Detect duplicate single-module alias bound to different modules.

    ``import foo as A`` + ``import bar as A`` → AglScopeError.
    Wildcard aliases are excluded here (their conflict is detected via
    ``_check_alias_root_collision`` and the qualified-merge conflict check).
    """
    # alias → (bound_module, first_decl)
    alias_module: dict[str, tuple[ModuleId, ImportDecl]] = {}

    for decl in decls:
        if decl.alias is None or decl.wildcard:
            continue
        # A non-wildcard import always resolves to a SingleTarget.
        target = targets[decl.node_id]
        assert isinstance(target, SingleTarget)
        mod = target.module
        if decl.alias in alias_module:
            bound_mod, _first_decl = alias_module[decl.alias]
            if bound_mod != mod:
                raise AglScopeError(
                    f"alias {decl.alias!r} is already bound to module"
                    f" {_module_path(bound_mod)!r};"
                    f" cannot rebind to {_module_path(mod)!r}",
                    span=decl.span,
                )
        else:
            alias_module[decl.alias] = (mod, decl)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def build_import_env(
    current_module: ModuleId,
    decls: tuple[ImportDecl, ...],
    targets: Mapping[int, ImportTarget],
    exports: Mapping[ModuleId, Mapping[str, "QName"]],
) -> ImportEnv:
    """Compute the :class:`ImportEnv` for *current_module* from its import declarations.

    Parameters
    ----------
    current_module:
        The :class:`ModuleId` of the module being resolved. It is accepted to
        keep this interface aligned with program-level import construction.
    decls:
        The module's import declarations in source order.
    targets:
        Maps each ``ImportDecl.node_id`` to its resolved :class:`ImportTarget`.
    exports:
        Maps each module referenced in *targets* to its export map: a dict of
        exported name → origin :data:`QName`.  For locally-defined names the
        origin is ``(module, name)``; for re-exported names it is the original
        defining module and name, preserving identity through re-export chains.

    Returns
    -------
    ImportEnv
        The frozen import environment.

    Raises
    ------
    AglScopeError
        On any static import conflict (duplicate alias, alias-root collision,
        ``using``/``hiding`` naming a non-exported name, or a qualified-handle
        conflict at merge time).
    """
    # Static checks that operate on the full decl list before per-decl processing.
    _check_duplicate_single_alias(decls, targets)
    _check_alias_root_collision(decls, targets)

    # Accumulate results.
    # unqualified: name → set of QName (mutable during build; frozen at the end)
    unqualified_acc: dict[str, set[QName]] = {}
    # qualified: handle → {name → QName}  (mutable dict of mutable dicts)
    qualified_acc: dict[tuple[str, ...], dict[str, QName]] = {}

    for decl in decls:
        target = targets[decl.node_id]

        if isinstance(target, SingleTarget):
            _process_single(decl, target.module, exports, unqualified_acc, qualified_acc)
        else:
            # WildcardTarget
            _process_wildcard(decl, target.modules, exports, unqualified_acc, qualified_acc)

    # Freeze and return.
    frozen_unqualified: dict[str, frozenset[QName]] = {
        name: frozenset(qnames) for name, qnames in unqualified_acc.items()
    }
    frozen_qualified: dict[tuple[str, ...], dict[str, QName]] = {
        handle: dict(name_map) for handle, name_map in qualified_acc.items()
    }
    return ImportEnv(unqualified=frozen_unqualified, qualified=frozen_qualified)


# ---------------------------------------------------------------------------
# Per-decl processing helpers
# ---------------------------------------------------------------------------


def _register_qname(
    handle: tuple[str, ...],
    exposed: str,
    qname: QName,
    qualified_acc: dict[tuple[str, ...], dict[str, QName]],
    decl: ImportDecl,
) -> None:
    """Register (handle, exposed) → qname in the qualified accumulator.

    Raises :class:`~agm.agl.scope.symbols.AglScopeError` if the handle+exposed
    key already maps to a *different* QName (static qualified-conflict).
    Idempotent when the existing mapping equals *qname*.
    """
    if handle not in qualified_acc:
        qualified_acc[handle] = {}
    existing = qualified_acc[handle].get(exposed)
    if existing is not None and existing != qname:
        raise AglScopeError(
            f"qualifier {'.'.join(handle)!r} maps name {exposed!r}"
            f" to both {_module_path(existing[0])!r} and"
            f" {_module_path(qname[0])!r}",
            span=decl.span,
        )
    qualified_acc[handle][exposed] = qname


def _process_single(
    decl: ImportDecl,
    module: ModuleId,
    exports: Mapping[ModuleId, Mapping[str, "QName"]],
    unqualified_acc: dict[str, set[QName]],
    qualified_acc: dict[tuple[str, ...], dict[str, QName]],
) -> None:
    """Process a single-module ImportDecl, updating the accumulators."""
    module_exports_map: Mapping[str, QName] = exports.get(module, {})
    module_exports: frozenset[str] = frozenset(module_exports_map.keys())
    s = _compute_s(decl, module, module_exports, is_wildcard=False)
    rename_map = _build_rename_map(decl)
    handle = _qualifier_handle(decl, module)

    for src_name in sorted(s):  # deterministic order
        exposed = _exposed_name(src_name, rename_map)
        qname: QName = module_exports_map[src_name]
        _register_qname(handle, exposed, qname, qualified_acc, decl)
        if not decl.qualified:
            if exposed not in unqualified_acc:
                unqualified_acc[exposed] = set()
            unqualified_acc[exposed].add(qname)


def _process_wildcard(
    decl: ImportDecl,
    modules: frozenset[ModuleId],
    exports: Mapping[ModuleId, Mapping[str, "QName"]],
    unqualified_acc: dict[str, set[QName]],
    qualified_acc: dict[tuple[str, ...], dict[str, QName]],
) -> None:
    """Process a wildcard ImportDecl, updating the accumulators."""
    # Build per-module name-sets and maps (only the matched modules)
    matched_export_names: dict[ModuleId, frozenset[str]] = {
        mod: frozenset(exports.get(mod, {}).keys()) for mod in modules
    }
    matched_export_maps: dict[ModuleId, Mapping[str, QName]] = {
        mod: exports.get(mod, {}) for mod in modules
    }

    # Validate using/hiding names against the union of all matched exports
    _validate_wildcard_items(decl, matched_export_names)

    rename_map = _build_rename_map(decl)

    # Process each matched module in deterministic order.
    for module in sorted(modules, key=_module_sort_key):
        module_export_names = matched_export_names[module]
        module_exports_map = matched_export_maps[module]
        s = _compute_s(decl, module, module_export_names, is_wildcard=True)
        handle = _qualifier_handle(decl, module)

        for src_name in sorted(s):  # deterministic order
            exposed = _exposed_name(src_name, rename_map)
            qname: QName = module_exports_map[src_name]
            _register_qname(handle, exposed, qname, qualified_acc, decl)
            if not decl.qualified:
                if exposed not in unqualified_acc:
                    unqualified_acc[exposed] = set()
                unqualified_acc[exposed].add(qname)


# ---------------------------------------------------------------------------
# Contribution import model
# ---------------------------------------------------------------------------


def _immutable_members(members: Mapping[str, QName]) -> Mapping[str, QName]:
    """Copy *members* into a deterministically ordered immutable mapping."""
    return MappingProxyType({name: qname for name, qname in sorted(members.items())})


def _immutable_unqualified(
    unqualified: Mapping[str, frozenset[QName]],
) -> Mapping[str, frozenset[QName]]:
    """Copy bare-name candidates into a deterministically ordered immutable mapping."""
    return MappingProxyType(
        {name: frozenset(qnames) for name, qnames in sorted(unqualified.items())}
    )


def _immutable_contributions(
    contributions: Mapping[ModuleId, "ModuleContribution"],
) -> Mapping[ModuleId, "ModuleContribution"]:
    """Copy module contributions into a deterministic immutable mapping."""
    return MappingProxyType(
        {
            module: contributions[module]
            for module in sorted(contributions, key=_module_sort_key)
        }
    )


def _immutable_names(names: frozenset[str]) -> frozenset[str]:
    """Copy a set of names into an immutable set."""
    return frozenset(names)


@dataclass(frozen=True, slots=True)
class ModuleContribution:
    """The merged contribution of one imported module.

    ``members`` is the module's unioned contributed set S, keyed by its
    canonical exposed names. ``bare_names`` is the subset injected into the
    unqualified namespace by ``using`` or an ``open`` import. ``path_enabled``
    records whether a plain import makes the module available to suffix and
    anchored qualification; aliases are instead listed in ``aliases``.
    """

    module: ModuleId
    members: Mapping[str, QName]
    bare_names: frozenset[str]
    path_enabled: bool
    aliases: frozenset[str]

    def __post_init__(self) -> None:
        members = _immutable_members(self.members)
        bare_names = _immutable_names(self.bare_names)
        aliases = _immutable_names(self.aliases)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "bare_names", bare_names)
        object.__setattr__(self, "aliases", aliases)


@dataclass(frozen=True, slots=True)
class ImportEnvV2:
    """Pure contribution import environment.

    Contributions are indexed by imported module so repeated declarations union
    their selected sets. The unqualified map retains lazy-clash candidates.
    """

    contributions: Mapping[ModuleId, ModuleContribution]
    unqualified: Mapping[str, frozenset[QName]]

    def __post_init__(self) -> None:
        contributions = _immutable_contributions(self.contributions)
        unqualified = _immutable_unqualified(self.unqualified)
        object.__setattr__(self, "contributions", contributions)
        object.__setattr__(self, "unqualified", unqualified)


@dataclass(frozen=True, slots=True)
class QualResolutionFound:
    """A qualified reference resolved to one contributed member."""

    module: ModuleId
    qname: QName


@dataclass(frozen=True, slots=True)
class QualResolutionUnknownQualifier:
    """No imported route matches the qualifier."""

    qualifier: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QualResolutionMissingMember:
    """The qualifier matches imports, but none contributes the requested member."""

    qualifier: tuple[str, ...]
    member: str
    candidates: tuple[ModuleId, ...]


@dataclass(frozen=True, slots=True)
class QualResolutionAmbiguous:
    """Several matching modules contribute the requested member."""

    qualifier: tuple[str, ...]
    member: str
    candidates: tuple[ModuleId, ...]


QualResolution = (
    QualResolutionFound
    | QualResolutionUnknownQualifier
    | QualResolutionMissingMember
    | QualResolutionAmbiguous
)


@dataclass(slots=True)
class _ContributionAccumulator:
    """Mutable builder state for one module's merged contribution."""

    members: dict[str, QName]
    bare_names: set[str]
    path_enabled: bool
    aliases: set[str]


def _compute_v2_set(
    decl: ImportDecl,
    module: ModuleId,
    module_exports: Mapping[str, QName],
) -> frozenset[str]:
    """Select one declaration's S for one target-module expansion.

    Wildcard imports expand per module, so a ``using`` or ``hiding`` item
    must be public in every expanded module just as it must be
    for a non-wildcard import.
    """
    exported_names = frozenset(module_exports)
    if decl.mode == ImportMode.ALL:
        return exported_names

    selected_names = frozenset(item.name for item in decl.items)
    for name in selected_names:
        if name not in exported_names:
            raise AglScopeError(
                f"name {name!r} is not exported by module {_module_path(module)!r}",
                span=decl.span,
            )
    if decl.mode == ImportMode.USING:
        return selected_names
    return exported_names - selected_names


def _v2_targets(target: ImportTarget) -> tuple[ModuleId, ...]:
    """Return a deterministic per-module expansion of an import target."""
    if isinstance(target, SingleTarget):
        return (target.module,)
    return tuple(sorted(target.modules, key=_module_sort_key))


def _add_v2_member(
    accumulator: _ContributionAccumulator,
    exposed_name: str,
    qname: QName,
    decl: ImportDecl,
) -> None:
    """Merge a canonical exposed member, rejecting distinct origins."""
    existing = accumulator.members.get(exposed_name)
    if existing is not None and existing != qname:
        raise AglScopeError(
            f"conflicting origins for exposed name {exposed_name!r}"
            f" ({_module_path(existing[0])!r} and {_module_path(qname[0])!r})",
            span=decl.span,
        )
    accumulator.members[exposed_name] = qname


def build_import_env_v2(
    decls: tuple[ImportDecl, ...],
    targets: Mapping[int, ImportTarget],
    exports: Mapping[ModuleId, Mapping[str, QName]],
    *,
    open_decl_ids: frozenset[int] = frozenset(),
) -> ImportEnvV2:
    """Build a contribution environment from resolved imports.

    ``open_decl_ids`` identifies declarations that inject their selected names
    into the bare namespace. A ``using`` list also injects its selected names.
    """
    accumulators: dict[ModuleId, _ContributionAccumulator] = {}

    for decl in decls:
        is_open = decl.node_id in open_decl_ids
        if is_open and decl.mode == ImportMode.USING:
            raise AglScopeError("open imports cannot also use a using clause", span=decl.span)

        for module in _v2_targets(targets[decl.node_id]):
            accumulator = accumulators.setdefault(
                module,
                _ContributionAccumulator(
                    members={}, bare_names=set(), path_enabled=False, aliases=set()
                ),
            )
            if decl.alias is None:
                accumulator.path_enabled = True
            else:
                accumulator.aliases.add(decl.alias)

            module_exports = exports.get(module, {})
            selected = _compute_v2_set(decl, module, module_exports)
            rename_map = _build_rename_map(decl)
            inject_bare = is_open or decl.mode == ImportMode.USING
            for source_name in sorted(selected):
                exposed_name = _exposed_name(source_name, rename_map)
                _add_v2_member(accumulator, exposed_name, module_exports[source_name], decl)
                if inject_bare:
                    accumulator.bare_names.add(exposed_name)

    contributions: dict[ModuleId, ModuleContribution] = {}
    unqualified_acc: dict[str, set[QName]] = {}
    for module in sorted(accumulators, key=_module_sort_key):
        accumulator = accumulators[module]
        contribution = ModuleContribution(
            module=module,
            members=dict(accumulator.members),
            bare_names=frozenset(accumulator.bare_names),
            path_enabled=accumulator.path_enabled,
            aliases=frozenset(accumulator.aliases),
        )
        contributions[module] = contribution
        for name in sorted(contribution.bare_names):
            unqualified_acc.setdefault(name, set()).add(contribution.members[name])

    return ImportEnvV2(
        contributions=contributions,
        unqualified={
            name: frozenset(unqualified_acc[name]) for name in sorted(unqualified_acc)
        },
    )


def _v2_qualifier_candidates(
    env: ImportEnvV2,
    qualifier: tuple[str, ...],
    *,
    anchored: bool,
) -> tuple[ModuleId, ...]:
    """Collect every route matching a qualifier, without applying precedence."""
    candidates: set[ModuleId] = set()
    for module, contribution in env.contributions.items():
        matches_path = contribution.path_enabled and (
            module.segments == qualifier
            if anchored
            else bool(qualifier) and module.segments[-len(qualifier) :] == qualifier
        )
        matches_alias = (
            not anchored and len(qualifier) == 1 and qualifier[0] in contribution.aliases
        )
        if matches_path or matches_alias:
            candidates.add(module)
    return tuple(sorted(candidates, key=_module_sort_key))


def resolve_qualified(
    env: ImportEnvV2,
    qualifier: tuple[str, ...],
    member: str,
    *,
    anchored: bool = False,
) -> QualResolution:
    """Resolve one qualified reference with suffix/anchor semantics.

    Routes are collected before their members are considered. Member filtering
    can therefore rescue a shared suffix or alias facade, while multiple
    surviving candidates always remain an ambiguity: there is no route, suffix,
    alias, or declaration-order preference.
    """
    candidates = _v2_qualifier_candidates(env, qualifier, anchored=anchored)
    if not candidates:
        return QualResolutionUnknownQualifier(qualifier)

    member_candidates = tuple(
        module for module in candidates if member in env.contributions[module].members
    )
    if not member_candidates:
        return QualResolutionMissingMember(qualifier, member, candidates)
    if len(member_candidates) > 1:
        return QualResolutionAmbiguous(qualifier, member, member_candidates)

    module = member_candidates[0]
    return QualResolutionFound(module, env.contributions[module].members[member])
