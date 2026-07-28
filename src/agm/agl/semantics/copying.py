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
from typing import assert_never

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
    shell = _empty_shell(value)
    match value, shell:
        case ArrayValue(elements=elements), ArrayValue():
            shell.elements.extend(elements)
        case DictValue(entries=entries), DictValue():
            shell.entries.update(entries)
        case RecordValue(fields=fields), RecordValue():
            shell.fields.update(fields)
        case EnumValue(fields=fields), EnumValue():
            shell.fields.update(fields)
        case ExceptionValue(fields=fields), ExceptionValue():
            shell.fields.update(fields)
        case _ as unreachable:  # pragma: no cover
            raise AssertionError(f"unexpected copy shell pair: {unreachable!r}")
    return shell


def deep_copy_value(value: Value) -> Value:
    """Return a fully independent deep copy of *value*, preserving sharing."""
    return _deep_copy(value, {})


#: The value kinds :func:`_deep_copy` rebuilds. Everything else — scalars,
#: ``unit``, agents, constructors, closures — is returned as-is and never
#: enters the memo, so copying an array of scalars costs no lookups.
_SHELL_KINDS = (ArrayValue, DictValue, RecordValue, EnumValue, ExceptionValue)
_COPIED_KINDS = (*_SHELL_KINDS, JsonValue)


def _empty_shell(value: ArrayValue | DictValue | RecordValue | EnumValue | ExceptionValue) -> Value:
    """Build an empty mutable-payload shell of the same container kind as *value*."""
    match value:
        case ArrayValue():
            return ArrayValue(elements=[])
        case DictValue():
            return DictValue(entries={})
        case RecordValue(nominal=nominal, display_name=display_name):
            return RecordValue(nominal=nominal, display_name=display_name, fields={})
        case EnumValue(nominal=nominal, display_name=display_name, variant=variant):
            return EnumValue(nominal=nominal, display_name=display_name, variant=variant, fields={})
        case ExceptionValue(nominal=nominal, display_name=display_name):
            return ExceptionValue(nominal=nominal, display_name=display_name, fields={})
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _fill_field_dict(dest: dict[str, Value], src: dict[str, Value], memo: dict[int, Value]) -> None:
    """Populate *dest* with a deep copy of every entry in *src*."""
    for k, v in src.items():
        dest[k] = _deep_copy(v, memo)


def _deep_copy(value: Value, memo: dict[int, Value]) -> Value:
    """Copy *value*, reusing *memo* so shared references stay shared.

    Every arm follows the same three steps: build an EMPTY shell, register it
    in *memo*, then fill it. Registering before filling is what makes a
    self-referential array or dict terminate — the re-entrant reference
    resolves to the shell already under construction. (``json`` is atomic, so
    its arm builds the copy outright; it is still memoized so two references
    to one ``json`` leaf copy to one new leaf.)
    """
    if not isinstance(value, _COPIED_KINDS):
        return value
    cached = memo.get(id(value))
    if cached is not None:
        return cached
    if isinstance(value, _SHELL_KINDS):
        shell = _empty_shell(value)
        memo[id(value)] = shell
        match value, shell:
            case ArrayValue(elements=elements), ArrayValue():
                shell.elements.extend(_deep_copy(element, memo) for element in elements)
            case DictValue(entries=entries), DictValue():
                _fill_field_dict(shell.entries, entries, memo)
            case RecordValue(fields=fields), RecordValue():
                _fill_field_dict(shell.fields, fields, memo)
            case EnumValue(fields=fields), EnumValue():
                _fill_field_dict(shell.fields, fields, memo)
            case ExceptionValue(fields=fields), ExceptionValue():
                _fill_field_dict(shell.fields, fields, memo)
            case _ as unreachable:  # pragma: no cover
                raise AssertionError(f"unexpected copy shell pair: {unreachable!r}")
        return shell
    json_copy = JsonValue(_stdlib_copy.deepcopy(value.raw))
    memo[id(value)] = json_copy
    return json_copy
