"""Cross-member declaration validation shared by module and program checking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from agm.agl.modules.ids import ModuleId, spell_declaration
from agm.agl.scope.symbols import ModuleResolution
from agm.agl.semantics.type_table import DeclId, TypeTable, qualified_decl_name
from agm.agl.syntax.nodes import (
    EnumDef,
    ExceptionDef,
    FuncDef,
    RecordDef,
    TypeAlias,
    static_function_items,
    static_items,
    static_type_items,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck.env import AglTypeError


@dataclass(frozen=True, slots=True)
class _MemberDeclaration:
    """A source member declaration whose name occupies a nominal member namespace."""

    kind: Literal["field", "method"]
    span: SourceSpan | None


@dataclass(slots=True)
class _MemberIndex:
    """Every nominal member namespace this validation can reach.

    ``members`` is the single source for the question "what does this owner
    declare?": source declarations carry their spans, and an owner outside this
    compile unit — an imported base, or a type retained from an earlier REPL
    entry — contributes its registered fields and directly declared methods
    without one. ``declared`` names the owners this compile unit contributes
    to, which are exactly the owners whose members need checking.
    """

    members: dict[DeclId, dict[str, list[_MemberDeclaration]]] = field(default_factory=dict)
    declared: set[DeclId] = field(default_factory=set)

    def members_of(self, owner_id: DeclId) -> Mapping[str, list[_MemberDeclaration]]:
        """Return *owner_id*'s member namespace, empty for an owner never indexed."""
        return self.members.get(owner_id, {})


def _index_registered_owner(index: _MemberIndex, type_table: TypeTable, owner_id: DeclId) -> None:
    """Index an owner outside this compile unit from its registered declaration."""
    if owner_id in index.members:
        return
    typedef = type_table.get_by_id(owner_id)
    assert typedef is not None, "compiler bug: related owner is not registered"
    members = index.members.setdefault(owner_id, {})
    for field_name, _field_type in typedef.fields:
        members.setdefault(field_name, []).append(_MemberDeclaration("field", None))
    for method_name in type_table.declared_methods(owner_id):
        members.setdefault(method_name, []).append(_MemberDeclaration("method", None))


def _member_declarations(
    modules: Mapping[ModuleId, ModuleResolution], type_table: TypeTable
) -> _MemberIndex:
    """Collect source and retained-owner fields plus scope-classified methods."""
    index = _MemberIndex()
    owner_ids: dict[tuple[ModuleId, tuple[str, ...]], DeclId] = {}
    for module_id, resolved in modules.items():
        for item in static_type_items(resolved.program.body.items):
            if isinstance(item, TypeAlias):
                continue
            declared_path = tuple(segment.name for segment in item.scope_path)
            typedef = type_table.get(module_id, item.name, declared_path)
            assert typedef is not None, "compiler bug: declared type is not registered"
            decl_owner_id = typedef.decl_node_id
            owner_ids[module_id, (*declared_path, item.name)] = decl_owner_id
            index.declared.add(decl_owner_id)
            members = index.members.setdefault(decl_owner_id, {})
            if isinstance(item, (RecordDef, ExceptionDef)):
                for source_field in item.fields:
                    members.setdefault(source_field.name, []).append(
                        _MemberDeclaration("field", source_field.span)
                    )
        for function in static_function_items(resolved.program.body.items):
            owner_path = resolved.receiver_owner_for(module_id, function)
            if owner_path is None:
                continue
            method_owner_id = owner_ids.get((module_id, owner_path))
            if method_owner_id is None:
                # A method on an owner retained from an earlier REPL entry: its
                # members come from the registry, without source spans.
                typedef = type_table.get(module_id, owner_path[-1], owner_path[:-1])
                assert typedef is not None, "compiler bug: method owner is not registered"
                method_owner_id = typedef.decl_node_id
                owner_ids[module_id, owner_path] = method_owner_id
                _index_registered_owner(index, type_table, method_owner_id)
            index.declared.add(method_owner_id)
            same_named = index.members.setdefault(method_owner_id, {}).setdefault(function.name, [])
            # This declaration supersedes its own earlier registration on a
            # retained owner; the owner's fields stay, since only a
            # redeclaration of the type itself can replace those.
            same_named[:] = [
                member
                for member in same_named
                if not (member.kind == "method" and member.span is None)
            ]
            same_named.append(_MemberDeclaration("method", function.span))
    return index


def _descendant_index(type_table: TypeTable) -> dict[DeclId, tuple[DeclId, ...]]:
    """Map every registered exception to its descendants, nearest first.

    The inverse of ``TypeTable.ancestor_defs``: a base declaration needs it to
    see the members its descendants already occupy, including descendants
    retained from an earlier REPL entry. Inverting the relation once per
    validation pass keeps the whole pass linear in the registered exceptions;
    asking each owner for its own descendants separately would rescan every
    declaration and rewalk every base chain per owner.

    A descendant's depth is its own index within its nearest-first ancestor
    chain, and ``entries()`` yields declarations in registration order, so
    descendants at equal depth keep a deterministic relative order.
    """
    found: dict[DeclId, list[tuple[int, DeclId]]] = {}
    for typedef in type_table.entries():
        if typedef.kind != "exception":
            continue
        candidate_id = typedef.decl_node_id
        if type_table.is_orphaned(candidate_id):
            # A declaration an incremental entry never promoted: it has no
            # values and no name, so its members constrain nothing.
            continue
        for depth, base in enumerate(type_table.ancestor_defs(candidate_id)):
            found.setdefault(base.decl_node_id, []).append((depth, candidate_id))
    result: dict[DeclId, tuple[DeclId, ...]] = {}
    for base_id, entries in found.items():
        entries.sort(key=_descendant_entry_depth)
        result[base_id] = tuple(candidate for _depth, candidate in entries)
    return result


def _descendant_entry_depth(entry: tuple[int, DeclId]) -> int:
    """Order a ``_descendant_index`` entry by its distance from the queried base."""
    return entry[0]


def _relatives(
    index: _MemberIndex,
    type_table: TypeTable,
    descendant_index: Mapping[DeclId, tuple[DeclId, ...]],
    owner_id: DeclId,
) -> tuple[
    tuple[DeclId, ...],
    tuple[DeclId, ...],
]:
    """Return *owner_id*'s indexed exception ancestors and unchecked descendants.

    A descendant this compile unit also declares is skipped: its own ancestor
    scan already covers the pair, so reporting it from both ends would make the
    diagnostic depend on iteration order.
    """
    ancestors = tuple(base.decl_node_id for base in type_table.ancestor_defs(owner_id))
    descendants = tuple(
        descendant_id
        for descendant_id in descendant_index.get(owner_id, ())
        if descendant_id not in index.declared
    )
    for relative in (*ancestors, *descendants):
        _index_registered_owner(index, type_table, relative)
    return ancestors, descendants


def _related_member(
    index: _MemberIndex, related: tuple[DeclId, ...], name: str
) -> tuple[DeclId, _MemberDeclaration] | None:
    """Find the nearest related owner that already declares *name*."""
    for relative in related:
        members = index.members_of(relative).get(name)
        if members:
            return relative, members[0]
    return None


def _raise_collision(
    type_table: TypeTable,
    owner_id: DeclId,
    name: str,
    declared: _MemberDeclaration,
    conflicting_id: DeclId,
    conflicting: _MemberDeclaration,
) -> None:
    """Report the later direct declaration or the more-specific descendant."""
    if owner_id == conflicting_id:
        # A retained owner's registration supplies members without source
        # spans. Its later-entry method is necessarily the one to diagnose.
        if declared.span is None:
            assert conflicting.span is not None
            declared, conflicting = conflicting, declared
        else:
            assert conflicting.span is not None
            if declared.span.start_offset < conflicting.span.start_offset:
                declared, conflicting = conflicting, declared

    owner_typedef = type_table.get_by_id(owner_id)
    conflicting_typedef = type_table.get_by_id(conflicting_id)
    assert owner_typedef is not None and conflicting_typedef is not None, (
        "compiler bug: collision owner is not registered"
    )
    related = (
        ()
        if conflicting.span is None
        else ((f"{conflicting.kind} '{name}' is declared here", conflicting.span),)
    )
    raise AglTypeError(
        f"{declared.kind.capitalize()} '{qualified_decl_name(owner_typedef)}::{name}' conflicts "
        f"with {conflicting.kind} '{name}' of '{qualified_decl_name(conflicting_typedef)}'.",
        span=declared.span,
        related=related,
    )


def _module_visit_order(module_id: ModuleId) -> tuple[bool, tuple[str, ...]]:
    """Sort key placing the entry module after every other module.

    A duplicate between a library declaration and the program's own is then
    reported at the program's own, more actionable, declaration.
    """
    return (module_id.is_entry, module_id.segments)


def _builtin_bare_declarations(
    modules: Mapping[ModuleId, ModuleResolution],
) -> list[tuple[ModuleId, tuple[str, ...], str, SourceSpan]]:
    """List every builtin type and builtin def declaration by scoped name.

    A ``builtin`` type (record/enum/exception) and a ``builtin def`` share one
    name space within the declaration's scope path. A ``builtin var`` is
    neither, and is excluded. This lets a builtin method coexist with the
    root builtin it dispatches to while retaining one declaration per scoped
    host identity.

    Modules are visited in :func:`_module_visit_order`, and each module's
    items in source order (named scope regions inlined), so a program with
    several duplicate declarations always reports the same one.
    """
    found: list[tuple[ModuleId, tuple[str, ...], str, SourceSpan]] = []
    for module_id in sorted(modules, key=_module_visit_order):
        resolved = modules[module_id]
        for item in static_items(resolved.program.body.items):
            if not isinstance(item, (RecordDef, EnumDef, ExceptionDef, FuncDef)):
                continue
            if not item.is_builtin:
                continue
            path = tuple(segment.name for segment in item.scope_path)
            found.append((module_id, path, item.name, item.span))
    return found


def validate_builtin_declaration_uniqueness(
    modules: Mapping[ModuleId, ModuleResolution],
) -> None:
    """Reject a builtin name declared more than once at the same scope path.

    A ``builtin`` declaration marks what a declaration denotes at the host
    boundary. Its scoped name, rather than its bare spelling, is the identity:
    a root builtin and a method under a receiver type intentionally coexist.
    Types and defs still share one namespace at any one scoped name.
    """
    first_seen: dict[tuple[str, ...], tuple[ModuleId, tuple[str, ...], SourceSpan]] = {}
    for module_id, path, name, span in _builtin_bare_declarations(modules):
        key = (*path, name)
        prior = first_seen.get(key)
        if prior is None:
            first_seen[key] = (module_id, path, span)
            continue
        prior_module_id, prior_path, prior_span = prior
        first_spelling = spell_declaration(prior_module_id, (*prior_path, name))
        duplicate_spelling = spell_declaration(module_id, (*path, name))
        raise AglTypeError(
            f"Builtin '{name}' is declared more than once: as '{first_spelling}' and as "
            f"'{duplicate_spelling}'.",
            span=span,
            related=((f"'{first_spelling}' is declared here", prior_span),),
        )


def validate_method_declaration_collisions(
    modules: Mapping[ModuleId, ModuleResolution], type_table: TypeTable
) -> None:
    """Reject field/method namespace collisions after nominal shapes are available.

    The scope pass is the authoritative source for method declarations, so this
    validation is independent of whether a method needs candidate inference
    before its header can be registered. An exception is checked against every
    ancestor and against every descendant this compile unit does not itself
    declare, so a base declared after its descendants is rejected too.
    """
    index = _member_declarations(modules, type_table)
    descendant_index = _descendant_index(type_table)

    def _sort_key(owner_id: DeclId) -> tuple[tuple[str, ...], tuple[str, ...], str]:
        return _owner_sort_key(type_table, owner_id)

    for owner_id in sorted(index.declared, key=_sort_key):
        ancestors, descendants = _relatives(index, type_table, descendant_index, owner_id)
        for name, same_named_members in index.members[owner_id].items():
            if len(same_named_members) > 1:
                _raise_collision(
                    type_table,
                    owner_id,
                    name,
                    same_named_members[0],
                    owner_id,
                    same_named_members[1],
                )

            member = same_named_members[0]
            for related in (ancestors, descendants):
                conflict = _related_member(index, related, name)
                if conflict is not None:
                    _raise_collision(type_table, owner_id, name, member, *conflict)


def _owner_sort_key(
    type_table: TypeTable, owner_id: DeclId
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Order owners deterministically so a program reports one stable collision."""
    typedef = type_table.get_by_id(owner_id)
    assert typedef is not None, "compiler bug: collision owner is not registered"
    return (typedef.module_id.segments, typedef.scope_path, typedef.name)
