"""Cross-member declaration validation shared by module and program checking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from agm.agl.modules.ids import PRELUDE_ID, ModuleId
from agm.agl.scope.symbols import ModuleResolution
from agm.agl.semantics.type_table import DeclKey, TypeTable, qualified_decl_name
from agm.agl.semantics.types import ExceptionType
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


def _member_declarations(
    modules: Mapping[ModuleId, ModuleResolution], type_table: TypeTable
) -> dict[DeclKey, dict[str, list[_MemberDeclaration]]]:
    """Collect source and retained-owner fields plus scope-classified methods."""
    result: dict[DeclKey, dict[str, list[_MemberDeclaration]]] = {}
    owner_keys: dict[tuple[ModuleId, tuple[str, ...]], DeclKey] = {}
    for module_id, resolved in modules.items():
        for item in static_type_items(resolved.program.body.items):
            if isinstance(item, TypeAlias):
                continue
            scope_path = tuple(segment.name for segment in item.scope_path)
            key = (PRELUDE_ID if item.is_builtin else module_id, scope_path, item.name)
            owner_keys[module_id, (*scope_path, item.name)] = key
            members = result.setdefault(key, {})
            if isinstance(item, (RecordDef, ExceptionDef)):
                for field in item.fields:
                    members.setdefault(field.name, []).append(
                        _MemberDeclaration("field", field.span)
                    )
        for function in static_function_items(resolved.program.body.items):
            owner_path = resolved.receiver_owner_for(module_id, function)
            if owner_path is None:
                continue
            owner_key = owner_keys.get((module_id, owner_path))
            if owner_key is None:
                # A method on an owner retained from an earlier REPL entry: its
                # fields come from the registered TypeDef, without source spans.
                typedef = type_table.get(module_id, owner_path[-1], owner_path[:-1])
                assert typedef is not None, "compiler bug: method owner is not registered"
                owner_key = (typedef.module_id, typedef.scope_path, typedef.name)
                owner_keys[module_id, owner_path] = owner_key
                owner_members = result.setdefault(owner_key, {})
                for field_name, _field_type in typedef.fields:
                    owner_members.setdefault(field_name, [_MemberDeclaration("field", None)])
            result.setdefault(owner_key, {}).setdefault(function.name, []).append(
                _MemberDeclaration("method", function.span)
            )
    return result


def _ancestors(type_table: TypeTable, key: DeclKey) -> tuple[tuple[DeclKey, frozenset[str]], ...]:
    """Pair each of *key*'s exception ancestors with its own field names.

    Empty for any owner that is not an exception descendant, so a record or enum
    needs no per-member inheritance scan at all.
    """
    return tuple(
        (
            (base.module_id, base.scope_path, base.name),
            frozenset(field_name for field_name, _type in base.fields),
        )
        for base in type_table.ancestor_defs(key)
    )


def _inherited_member(
    type_table: TypeTable,
    declarations: Mapping[DeclKey, Mapping[str, list[_MemberDeclaration]]],
    ancestors: tuple[tuple[DeclKey, frozenset[str]], ...],
    name: str,
) -> tuple[DeclKey, _MemberDeclaration] | None:
    """Find the nearest ancestor member named *name*."""
    for base, base_fields in ancestors:
        members = declarations.get(base, {}).get(name)
        if members:
            return base, members[0]
        if name in base_fields:
            return base, _MemberDeclaration("field", None)
        base_handle = ExceptionType(name=base[2], module_id=base[0], scope_path=base[1])
        if type_table.lookup_method(base_handle, name) is not None:
            return base, _MemberDeclaration("method", None)
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
        # A retained owner's TypeDef supplies fields without source spans. Its
        # later-entry method is necessarily the declaration to diagnose.
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
    before its header can be registered. Exception descendants are checked
    against every ancestor, with the descendant declaration receiving the
    diagnostic.
    """
    declarations = _member_declarations(modules, type_table)
    for key, members in declarations.items():
        ancestors = _ancestors(type_table, key)
        for name, same_named_members in members.items():
            if len(same_named_members) > 1:
                _raise_collision(key, name, same_named_members[0], key, same_named_members[1])

            member = same_named_members[0]
            inherited = _inherited_member(type_table, declarations, ancestors, name)
            if inherited is not None:
                inherited_key, inherited_member = inherited
                _raise_collision(key, name, member, inherited_key, inherited_member)
