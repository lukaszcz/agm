"""Shared ``copy``/``shallow_copy`` value walks for the AgL runtime.

Single implementation of the two copy builtins, so the IR interpreter only
evaluates the argument and dispatches here.

``shallow_copy_value`` rebuilds exactly one container level — a fresh array,
dict, record, enum, or exception holding the *same* element/field
references as the original. It never recurses, so it can never loop and
never raises, even on a cyclic value.

``deep_copy_value`` recurses through every array, dict, record, enum, and
exception reachable from the value, rebuilding each with independently
copied contents. An ``id()``-keyed memo makes two references to the same
object copy to the same new object: a diamond stays a diamond, and — because
every container-shaped value is registered in the memo *before* its contents
are copied — a self-referential array or dict (a genuine reference cycle)
resolves the re-entrant reference to the still-being-built copy instead of
recursing forever. This is the one deep, structure-rebuilding walk in the
language that traverses a cyclic value to completion, terminating with an
isomorphic independent cycle instead of raising ``CyclicValueError``.

Both copies share one container walk, :func:`_copy_container`, which switches
once on the container kind, builds an empty shell, hands it to a *register*
callback, and only then fills each child. Filling takes an optional
*transform*: ``None`` means "copy the children as-is", filled in bulk via
``list.extend``/``dict.update`` at C speed; a callback means a per-child
copy, applied one element/entry at a time. ``shallow_copy_value`` passes
``transform=None`` — a shallow copy never recurses, so its children are
copied as-is and it pays no per-element Python call. ``deep_copy_value``
passes a *transform* that recurses and memoizes. Invoking *register* between
construction and filling — rather than after — structurally guarantees the
before-children registration order described above: ``shallow_copy_value``
also passes a no-op *register* (it never recurses, so there is nothing to
memoize), while ``deep_copy_value`` passes a *register* that writes the
shell into the memo.

A ``json`` leaf is duplicated via ``copy.deepcopy`` (its payload is a plain,
already-acyclic Python tree — see ``semantics/values.py``'s ``JsonValue``) and
is entered in the same memo as every other kind, so two references to the
same ``json`` leaf copy to the same new leaf. Every other value kind —
scalars, ``unit``, agents, constructors, closures — is returned as-is:
primitives are immutable so there is nothing to detach, and the opaque kinds
are capability handles, not data.
"""

from __future__ import annotations

import copy as _stdlib_copy
from collections.abc import Callable

from agm.agl.semantics.values import (
    ArrayValue,
    DictValue,
    EnumValue,
    ExceptionValue,
    JsonValue,
    RecordValue,
    Value,
)

__all__ = ["deep_copy_value", "shallow_copy_value"]


def shallow_copy_value(value: Value) -> Value:
    """Return a one-level copy of *value*: same nested references, new container."""
    if not isinstance(value, _SHELL_KINDS):
        return value
    return _copy_container(value, transform=None, register=lambda _shell: None)


def deep_copy_value(value: Value) -> Value:
    """Return a fully independent deep copy of *value*, preserving sharing."""
    return _deep_copy(value, {})


#: The value kinds :func:`_deep_copy` rebuilds. Everything else — scalars,
#: ``unit``, agents, constructors, closures — is returned as-is and never
#: enters the memo, so copying an array of scalars costs no lookups.
_SHELL_KINDS = (ArrayValue, DictValue, RecordValue, EnumValue, ExceptionValue)
_COPIED_KINDS = (*_SHELL_KINDS, JsonValue)


def _extend_array(
    shell: ArrayValue, elements: list[Value], transform: Callable[[Value], Value] | None
) -> None:
    """Fill *shell* with *elements*, in bulk if *transform* is ``None``, else one by one."""
    if transform is None:
        shell.elements.extend(elements)
    else:
        shell.elements.extend(transform(element) for element in elements)


def _update_fields(
    shell_fields: dict[str, Value],
    fields: dict[str, Value],
    transform: Callable[[Value], Value] | None,
) -> None:
    """Fill *shell_fields* from *fields*, in bulk if *transform* is ``None``, else one by one."""
    if transform is None:
        shell_fields.update(fields)
    else:
        for key, field_value in fields.items():
            shell_fields[key] = transform(field_value)


def _copy_container(
    value: ArrayValue | DictValue | RecordValue | EnumValue | ExceptionValue,
    transform: Callable[[Value], Value] | None,
    register: Callable[[Value], None],
) -> Value:
    """Build an empty shell of *value*'s kind, register it, then fill it.

    *register* runs on the fresh, still-empty shell before any child is visited,
    so a caller that memoizes the shell there (as :func:`_deep_copy` does) is
    structurally guaranteed to have it in place before a re-entrant reference to
    *value* can be looked up — the ordering that lets a self-referential
    container terminate instead of recursing forever. *transform*, if given, is
    applied to each child value (a memoized recursive copy, for a deep copy);
    ``None`` fills the shell with the original children in one bulk call, for a
    shallow copy that never recurses.
    """
    if isinstance(value, ArrayValue):
        array_shell = ArrayValue(elements=[])
        register(array_shell)
        _extend_array(array_shell, value.elements, transform)
        return array_shell
    if isinstance(value, DictValue):
        dict_shell = DictValue(entries={})
        register(dict_shell)
        _update_fields(dict_shell.entries, value.entries, transform)
        return dict_shell
    if isinstance(value, RecordValue):
        record_shell = RecordValue(
            nominal=value.nominal, display_name=value.display_name, fields={}
        )
        register(record_shell)
        _update_fields(record_shell.fields, value.fields, transform)
        return record_shell
    if isinstance(value, EnumValue):
        enum_shell = EnumValue(
            nominal=value.nominal,
            display_name=value.display_name,
            variant=value.variant,
            fields={},
        )
        register(enum_shell)
        _update_fields(enum_shell.fields, value.fields, transform)
        return enum_shell
    exception_shell = ExceptionValue(
        nominal=value.nominal, display_name=value.display_name, fields={}
    )
    register(exception_shell)
    _update_fields(exception_shell.fields, value.fields, transform)
    return exception_shell


def _deep_copy(value: Value, memo: dict[int, Value]) -> Value:
    """Copy *value*, reusing *memo* so shared references stay shared.

    Every container arm goes through :func:`_copy_container`, which registers
    the empty shell in *memo* before filling it — that is what makes a
    self-referential array or dict terminate: the re-entrant reference resolves
    to the shell already under construction. (``json`` is atomic, so its arm
    builds the copy outright; it is still memoized so two references to one
    ``json`` leaf copy to one new leaf.)
    """
    if not isinstance(value, _COPIED_KINDS):
        return value
    cached = memo.get(id(value))
    if cached is not None:
        return cached
    if isinstance(value, _SHELL_KINDS):

        def register(shell: Value) -> None:
            memo[id(value)] = shell

        return _copy_container(value, transform=lambda v: _deep_copy(v, memo), register=register)
    json_copy = JsonValue(_stdlib_copy.deepcopy(value.raw))
    memo[id(value)] = json_copy
    return json_copy
