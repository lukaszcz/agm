"""Cross-member declaration validation shared by module and program checking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from agm.agl.modules.ids import PRELUDE_ID, ModuleId
from agm.agl.scope.symbols import ModuleResolution
from agm.agl.semantics.type_table import DeclKey, TypeTable, qualified_decl_name
from agm.agl.syntax.nodes import (
    ExceptionDef,
    RecordDef,
    TypeAlias,
    static_function_items,
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

    members: dict[DeclKey, dict[str, list[_MemberDeclaration]]] = field(default_factory=dict)
    declared: set[DeclKey] = field(default_factory=set)

    def members_of(self, key: DeclKey) -> Mapping[str, list[_MemberDeclaration]]:
        """Return *key*'s member namespace, empty for an owner never indexed."""
        return self.members.get(key, {})


def _index_registered_owner(index: _MemberIndex, type_table: TypeTable, key: DeclKey) -> None:
    """Index an owner outside this compile unit from its registered declaration."""
    if key in index.members:
        return
    typedef = type_table.get(key[0], key[2], key[1])
    assert typedef is not None, "compiler bug: related owner is not registered"
    members = index.members.setdefault(key, {})
    for field_name, _field_type in typedef.fields:
        members.setdefault(field_name, []).append(_MemberDeclaration("field", None))
    for method_name in type_table.declared_methods(key):
        members.setdefault(method_name, []).append(_MemberDeclaration("method", None))


def _member_declarations(
    modules: Mapping[ModuleId, ModuleResolution], type_table: TypeTable
) -> _MemberIndex:
    """Collect source and retained-owner fields plus scope-classified methods."""
    index = _MemberIndex()
    owner_keys: dict[tuple[ModuleId, tuple[str, ...]], DeclKey] = {}
    for module_id, resolved in modules.items():
        for item in static_type_items(resolved.program.body.items):
            if isinstance(item, TypeAlias):
                continue
            scope_path = tuple(segment.name for segment in item.scope_path)
            key = (PRELUDE_ID if item.is_builtin else module_id, scope_path, item.name)
            owner_keys[module_id, (*scope_path, item.name)] = key
            index.declared.add(key)
            members = index.members.setdefault(key, {})
            if isinstance(item, (RecordDef, ExceptionDef)):
                for source_field in item.fields:
                    members.setdefault(source_field.name, []).append(
                        _MemberDeclaration("field", source_field.span)
                    )
        for function in static_function_items(resolved.program.body.items):
            owner_path = resolved.receiver_owner_for(module_id, function)
            if owner_path is None:
                continue
            owner_key = owner_keys.get((module_id, owner_path))
            if owner_key is None:
                # A method on an owner retained from an earlier REPL entry: its
                # members come from the registry, without source spans.
                typedef = type_table.get(module_id, owner_path[-1], owner_path[:-1])
                assert typedef is not None, "compiler bug: method owner is not registered"
                owner_key = (typedef.module_id, typedef.scope_path, typedef.name)
                owner_keys[module_id, owner_path] = owner_key
                _index_registered_owner(index, type_table, owner_key)
            index.declared.add(owner_key)
            same_named = index.members.setdefault(owner_key, {}).setdefault(function.name, [])
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


def _relatives(
    index: _MemberIndex, type_table: TypeTable, key: DeclKey
) -> tuple[
    tuple[DeclKey, ...],
    tuple[DeclKey, ...],
]:
    """Return *key*'s indexed exception ancestors and unchecked descendants.

    A descendant this compile unit also declares is skipped: its own ancestor
    scan already covers the pair, so reporting it from both ends would make the
    diagnostic depend on iteration order.
    """
    ancestors = tuple(
        (base.module_id, base.scope_path, base.name) for base in type_table.ancestor_defs(key)
    )
    descendants = tuple(
        descendant_key
        for descendant in type_table.descendant_defs(key)
        if (descendant_key := (descendant.module_id, descendant.scope_path, descendant.name))
        not in index.declared
    )
    for relative in (*ancestors, *descendants):
        _index_registered_owner(index, type_table, relative)
    return ancestors, descendants


def _related_member(
    index: _MemberIndex, related: tuple[DeclKey, ...], name: str
) -> tuple[DeclKey, _MemberDeclaration] | None:
    """Find the nearest related owner that already declares *name*."""
    for relative in related:
        members = index.members_of(relative).get(name)
        if members:
            return relative, members[0]
    return None


def _raise_collision(
    key: DeclKey,
    name: str,
    declared: _MemberDeclaration,
    conflicting_key: DeclKey,
    conflicting: _MemberDeclaration,
) -> None:
    """Report the later direct declaration or the more-specific descendant."""
    if key == conflicting_key:
        # A retained owner's registration supplies members without source
        # spans. Its later-entry method is necessarily the one to diagnose.
        if declared.span is None:
            assert conflicting.span is not None
            declared, conflicting = conflicting, declared
        else:
            assert conflicting.span is not None
            if declared.span.start_offset < conflicting.span.start_offset:
                declared, conflicting = conflicting, declared

    related = (
        ()
        if conflicting.span is None
        else ((f"{conflicting.kind} '{name}' is declared here", conflicting.span),)
    )
    raise AglTypeError(
        f"{declared.kind.capitalize()} '{qualified_decl_name(key)}::{name}' conflicts with "
        f"{conflicting.kind} '{name}' of '{qualified_decl_name(conflicting_key)}'.",
        span=declared.span,
        related=related,
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
    for key in sorted(index.declared, key=_owner_sort_key):
        ancestors, descendants = _relatives(index, type_table, key)
        for name, same_named_members in index.members[key].items():
            if len(same_named_members) > 1:
                _raise_collision(key, name, same_named_members[0], key, same_named_members[1])

            member = same_named_members[0]
            for related in (ancestors, descendants):
                conflict = _related_member(index, related, name)
                if conflict is not None:
                    _raise_collision(key, name, member, *conflict)


def _owner_sort_key(key: DeclKey) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Order owners deterministically so a program reports one stable collision."""
    return (key[0].segments, key[1], key[2])
