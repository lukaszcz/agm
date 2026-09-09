"""Cross-member declaration validation shared by module and program checking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

from agm.agl.modules.ids import ModuleId, spell_declaration
from agm.agl.scope.symbols import BUILTIN_METHOD_RECEIVER_NAMES, ModuleResolution
from agm.agl.semantics.type_table import (
    DeclId,
    TypeTable,
    decl_id_sort_key,
    qualified_decl_name,
)
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
from agm.agl.syntax.types import AppliedT, ArrayT, DictT, NameT
from agm.agl.typecheck.env import AglTypeError


@dataclass(frozen=True, slots=True)
class BuiltinMethodReceiver:
    """A validated builtin method receiver and its optional binding parameter."""

    name: str
    type_parameter: str | None = None


def builtin_method_receiver_for(
    function: FuncDef, owner_path: tuple[str, ...]
) -> BuiltinMethodReceiver | None:
    """Resolve a builtin receiver spelling, rejecting unsupported applied forms."""
    receiver = function.receiver_type
    if isinstance(receiver, ArrayT):
        if not isinstance(receiver.elem, NameT):
            raise AglTypeError(
                "Builtin method receivers must use their bare generic form.", span=receiver.span
            )
        return BuiltinMethodReceiver("array", receiver.elem.name)
    if isinstance(receiver, DictT):
        if not isinstance(receiver.value, NameT):
            raise AglTypeError(
                "Builtin method receivers must use their bare generic form.", span=receiver.span
            )
        return BuiltinMethodReceiver("dict", receiver.value.name)
    if isinstance(receiver, AppliedT):
        raise AglTypeError("Unknown builtin method receiver.", span=receiver.span)
    if receiver is None and len(owner_path) == 1 and owner_path[0] in BUILTIN_METHOD_RECEIVER_NAMES:
        return BuiltinMethodReceiver(owner_path[0])
    return None


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
                    same_named = members.setdefault(source_field.name, [])
                    same_named[:] = [
                        member
                        for member in same_named
                        if not (member.kind == "field" and member.span is None)
                    ]
                    same_named.append(_MemberDeclaration("field", source_field.span))
        for function in static_function_items(resolved.program.body.items):
            owner_path = resolved.receiver_owner_for(module_id, function)
            if (
                owner_path is None
                or builtin_method_receiver_for(function, owner_path.scope_path) is not None
            ):
                continue
            method_owner_id = owner_ids.get((owner_path.module_id, owner_path.scope_path))
            if method_owner_id is None:
                # A method on an owner retained from an earlier REPL entry or
                # another module: its members come from the shared type table,
                # without source spans.
                typedef = type_table.get(
                    owner_path.module_id, owner_path.scope_path[-1], owner_path.scope_path[:-1]
                )
                assert typedef is not None, "compiler bug: method owner is not registered"
                method_owner_id = typedef.decl_node_id
                owner_ids[owner_path.module_id, owner_path.scope_path] = method_owner_id
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


def _ancestor_field(
    index: _MemberIndex, ancestors: tuple[DeclId, ...], name: str
) -> tuple[DeclId, _MemberDeclaration] | None:
    """Find the nearest ancestor field named *name*."""
    for ancestor in ancestors:
        fields = index.members_of(ancestor).get(name, ())
        field = next((member for member in fields if member.kind == "field"), None)
        if field is not None:
            return ancestor, field
    return None


def _raise_collision(
    type_table: TypeTable,
    owner_id: DeclId,
    name: str,
    declared: _MemberDeclaration,
    conflicting_id: DeclId,
    conflicting: _MemberDeclaration,
) -> None:
    """Report a method collision, preferring the later same-source declaration."""
    if (
        owner_id == conflicting_id
        and declared.span is not None
        and conflicting.span is not None
        and declared.span.source == conflicting.span.source
        and declared.span.start_offset < conflicting.span.start_offset
    ):
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


def _module_visit_order(module_id: ModuleId, entry_id: ModuleId) -> tuple[bool, tuple[str, ...]]:
    """Sort key placing the program's entry module after every other module.

    A duplicate between a library declaration and the program's own is then
    reported at the program's own, more actionable, declaration.
    """
    return (module_id == entry_id, module_id.segments)


def _builtin_bare_declarations(
    modules: Mapping[ModuleId, ModuleResolution],
    entry_id: ModuleId,
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
    for module_id in sorted(modules, key=partial(_module_visit_order, entry_id=entry_id)):
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
    entry_id: ModuleId,
) -> None:
    """Reject a builtin name declared more than once at the same scope path.

    A ``builtin`` declaration marks what a declaration denotes at the host
    boundary. Its scoped name, rather than its bare spelling, is the identity:
    a root builtin and a method under a receiver type intentionally coexist.
    Types and defs still share one namespace at any one scoped name.
    """
    first_seen: dict[tuple[str, ...], tuple[ModuleId, tuple[str, ...], SourceSpan]] = {}
    for module_id, path, name, span in _builtin_bare_declarations(modules, entry_id):
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
    """Reject method collisions with fields of their owner or its ancestors."""
    index = _member_declarations(modules, type_table)

    # Order owners deterministically so a program reports one stable collision.
    sort_key = partial(decl_id_sort_key, type_table.defs)
    for owner_id in sorted(index.declared, key=sort_key):
        ancestors = tuple(base.decl_node_id for base in type_table.ancestor_defs(owner_id))
        for ancestor in ancestors:
            _index_registered_owner(index, type_table, ancestor)
        for name, same_named_members in index.members[owner_id].items():
            for method in (member for member in same_named_members if member.kind == "method"):
                field = next(
                    (member for member in same_named_members if member.kind == "field"), None
                )
                if field is not None:
                    _raise_collision(type_table, owner_id, name, method, owner_id, field)
                conflict = _ancestor_field(index, ancestors, name)
                if conflict is not None:
                    _raise_collision(type_table, owner_id, name, method, *conflict)
