"""Cross-member declaration validation shared by module and program checking."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

from agm.agl.modules.ids import PRELUDE_ID, ModuleId
from agm.agl.scope.symbols import ModuleResolution
from agm.agl.semantics.type_table import DeclKey, TypeTable
from agm.agl.semantics.types import ExceptionType
from agm.agl.syntax.nodes import EnumDef, ExceptionDef, FuncDef, RecordDef, ScopeRegion
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck.env import AglTypeError


@dataclass(frozen=True, slots=True)
class _MemberDeclaration:
    """A source member declaration whose name occupies a nominal member namespace."""

    kind: Literal["field", "method"]
    span: SourceSpan | None


def _static_type_items(
    items: tuple[object, ...],
) -> Iterator[RecordDef | EnumDef | ExceptionDef]:
    """Yield nominal declarations from named scope regions."""
    for item in items:
        if isinstance(item, ScopeRegion):
            yield from _static_type_items(item.items)
        elif isinstance(item, (RecordDef, EnumDef, ExceptionDef)):
            yield item


def _static_function_items(items: tuple[object, ...]) -> Iterator[FuncDef]:
    """Yield functions from named scope regions."""
    for item in items:
        if isinstance(item, ScopeRegion):
            yield from _static_function_items(item.items)
        elif isinstance(item, FuncDef):
            yield item


def _member_declarations(
    modules: Mapping[ModuleId, ModuleResolution],
) -> dict[DeclKey, dict[str, list[_MemberDeclaration]]]:
    """Collect source fields and scope-classified methods by nominal owner."""
    result: dict[DeclKey, dict[str, list[_MemberDeclaration]]] = {}
    owner_keys: dict[tuple[ModuleId, tuple[str, ...]], DeclKey] = {}
    for module_id, resolved in modules.items():
        for item in _static_type_items(resolved.program.body.items):
            scope_path = tuple(segment.name for segment in item.scope_path)
            key = (PRELUDE_ID if item.is_builtin else module_id, scope_path, item.name)
            owner_keys[module_id, (*scope_path, item.name)] = key
            members = result.setdefault(key, {})
            if isinstance(item, (RecordDef, ExceptionDef)):
                for field in item.fields:
                    members.setdefault(field.name, []).append(
                        _MemberDeclaration("field", field.span)
                    )
        for function in _static_function_items(resolved.program.body.items):
            scope_path = tuple(segment.name for segment in function.scope_path)
            owner_path = resolved.method_declarations.get((module_id, scope_path, function.name))
            if owner_path is not None:
                key = owner_keys[module_id, owner_path]
                result.setdefault(key, {}).setdefault(function.name, []).append(
                    _MemberDeclaration("method", function.span)
                )
    return result


def _inherited_member(
    type_table: TypeTable,
    declarations: Mapping[DeclKey, Mapping[str, list[_MemberDeclaration]]],
    key: DeclKey,
    name: str,
) -> tuple[DeclKey, _MemberDeclaration] | None:
    """Find the nearest source-declared ancestor member named *name*."""
    typedef = type_table.get(key[0], key[2], key[1])
    assert typedef is not None
    base = typedef.base if typedef.kind == "exception" else None
    while base is not None:
        members = declarations.get(base, {}).get(name)
        if members:
            return base, members[0]
        base_def = type_table.get(base[0], base[2], base[1])
        assert base_def is not None
        if name in dict(base_def.fields):
            return base, _MemberDeclaration("field", None)
        base_handle = ExceptionType(name=base[2], module_id=base[0], scope_path=base[1])
        if type_table.lookup_method(base_handle, name) is not None:
            return base, _MemberDeclaration("method", None)
        base = base_def.base if base_def.kind == "exception" else None
    return None


def _owner_name(key: DeclKey) -> str:
    """Render an owner key in source-facing nominal spelling."""
    return "::".join((*key[1], key[2]))


def _raise_collision(
    key: DeclKey,
    name: str,
    declared: _MemberDeclaration,
    conflicting_key: DeclKey,
    conflicting: _MemberDeclaration,
) -> None:
    """Report the later direct declaration or the more-specific descendant."""
    if key == conflicting_key:
        assert declared.span is not None and conflicting.span is not None
        if declared.span.start_offset < conflicting.span.start_offset:
            declared, conflicting = conflicting, declared

    related = (
        ()
        if conflicting.span is None
        else ((f"{conflicting.kind} '{name}' is declared here", conflicting.span),)
    )
    raise AglTypeError(
        f"{declared.kind.capitalize()} '{_owner_name(key)}::{name}' conflicts with "
        f"{conflicting.kind} '{name}' of '{_owner_name(conflicting_key)}'.",
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
    declarations = _member_declarations(modules)
    for key, members in declarations.items():
        for name, same_named_members in members.items():
            if len(same_named_members) > 1:
                _raise_collision(key, name, same_named_members[0], key, same_named_members[1])

            member = same_named_members[0]
            inherited = _inherited_member(type_table, declarations, key, name)
            if inherited is not None:
                inherited_key, inherited_member = inherited
                _raise_collision(key, name, member, inherited_key, inherited_member)
