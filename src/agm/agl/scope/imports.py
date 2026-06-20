"""Pure import-resolution model for the AgL module system (D3).

This module exposes a single pure function :func:`build_import_env` that
computes a module's :class:`ImportEnv` from its :class:`~agm.agl.syntax.nodes.ImportDecl`
declarations and pre-built per-module export sets.  It has **no dependency** on
the resolver or scope-tree; it depends only on AST node types, :class:`ModuleId`,
and :class:`AglScopeError`.

This isolation lets D3's combinatorial import-merge logic be tested exhaustively in
unit tests without instantiating a full resolver.

Design decisions implemented
-----------------------------
- **D3** — all import forms: bare/``as``/``qualified``/``qualified as``/
  ``using``/``using…as``/``hiding``/wildcard (``.*``)/wildcard ``as`` re-rooting.
- **D5** — ``private`` names are absent from ``exports``; ``build_import_env``
  never re-admits them (it operates exclusively on the supplied ``exports`` map).
- **D10** — unified namespace: any top-level non-private ``FuncDef``, ``RecordDef``,
  ``EnumDef``, or ``TypeAlias`` name may appear in ``using``/``hiding``.

The :class:`ImportEnv` produced here is consumed by M3b (graph resolver) and M4
(typecheck) so both of those layers agree on import semantics.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
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


def _module_dotted(mid: ModuleId) -> str:
    """Return the display dotted name of a module (for error messages)."""
    return mid.dotted()


def _compute_s(
    decl: ImportDecl,
    module: ModuleId,
    module_exports: frozenset[str],
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
                        f"name {item.name!r} is not exported by module"
                        f" {_module_dotted(module)!r}",
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
                    f"name {item.name!r} is not exported by module"
                    f" {_module_dotted(module)!r}",
                    span=decl.span,
                )
            hidden.add(item.name)
    return module_exports - hidden


def _validate_wildcard_items(
    decl: ImportDecl,
    all_exports: Mapping[ModuleId, frozenset[str]],
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

    Rules (D3):
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
    """Detect alias-root vs non-alias module-path root collision (D3 static error).

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
    root_entries: dict[str, list[tuple[frozenset[ModuleId], bool, ImportDecl]]] = defaultdict(
        list
    )

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
                    f" {_module_dotted(bound_mod)!r};"
                    f" cannot rebind to {_module_dotted(mod)!r}",
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
    exports: Mapping[ModuleId, frozenset[str]],
) -> ImportEnv:
    """Compute the :class:`ImportEnv` for *current_module* from its import declarations.

    Parameters
    ----------
    current_module:
        The :class:`ModuleId` of the module being resolved (not used internally;
        reserved for future extension).
    decls:
        The module's import declarations in source order.
    targets:
        Maps each ``ImportDecl.node_id`` to its resolved :class:`ImportTarget`.
    exports:
        Maps each module referenced in *targets* to its export set (non-private
        top-level ``def``/``record``/``enum``/``type`` names).

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
            _process_wildcard(
                decl, target.modules, exports, unqualified_acc, qualified_acc
            )

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
            f" to both {_module_dotted(existing[0])!r} and"
            f" {_module_dotted(qname[0])!r}",
            span=decl.span,
        )
    qualified_acc[handle][exposed] = qname


def _process_single(
    decl: ImportDecl,
    module: ModuleId,
    exports: Mapping[ModuleId, frozenset[str]],
    unqualified_acc: dict[str, set[QName]],
    qualified_acc: dict[tuple[str, ...], dict[str, QName]],
) -> None:
    """Process a single-module ImportDecl, updating the accumulators."""
    module_exports: frozenset[str] = exports.get(module, frozenset())
    s = _compute_s(decl, module, module_exports, is_wildcard=False)
    rename_map = _build_rename_map(decl)
    handle = _qualifier_handle(decl, module)

    for src_name in sorted(s):  # deterministic order
        exposed = _exposed_name(src_name, rename_map)
        qname: QName = (module, src_name)
        _register_qname(handle, exposed, qname, qualified_acc, decl)
        if not decl.qualified:
            if exposed not in unqualified_acc:
                unqualified_acc[exposed] = set()
            unqualified_acc[exposed].add(qname)


def _process_wildcard(
    decl: ImportDecl,
    modules: frozenset[ModuleId],
    exports: Mapping[ModuleId, frozenset[str]],
    unqualified_acc: dict[str, set[QName]],
    qualified_acc: dict[tuple[str, ...], dict[str, QName]],
) -> None:
    """Process a wildcard ImportDecl, updating the accumulators."""
    # Build per-module exports subset (only the matched modules)
    matched_exports: dict[ModuleId, frozenset[str]] = {
        mod: exports.get(mod, frozenset()) for mod in modules
    }

    # Validate using/hiding names against the union of all matched exports
    _validate_wildcard_items(decl, matched_exports)

    rename_map = _build_rename_map(decl)

    # Process each matched module (deterministic order)
    def _module_sort_key(m: ModuleId) -> tuple[str, ...]:
        return m.segments

    for module in sorted(modules, key=_module_sort_key):
        module_exports = matched_exports[module]
        s = _compute_s(decl, module, module_exports, is_wildcard=True)
        handle = _qualifier_handle(decl, module)

        for src_name in sorted(s):  # deterministic order
            exposed = _exposed_name(src_name, rename_map)
            qname: QName = (module, src_name)
            _register_qname(handle, exposed, qname, qualified_acc, decl)
            if not decl.qualified:
                if exposed not in unqualified_acc:
                    unqualified_acc[exposed] = set()
                unqualified_acc[exposed].add(qname)
