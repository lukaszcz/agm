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
from collections.abc import Callable
from typing import TypeVar

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

_ShellT = TypeVar("_ShellT", bound=Value)


def shallow_copy_value(value: Value) -> Value:
    """Return a one-level copy of *value*: same nested references, new container."""
    if isinstance(value, ArrayValue):
        return ArrayValue(elements=list(value.elements))
    if isinstance(value, DictValue):
        return DictValue(entries=dict(value.entries))
    if isinstance(value, RecordValue):
        return RecordValue(
            nominal=value.nominal, display_name=value.display_name, fields=dict(value.fields)
        )
    if isinstance(value, EnumValue):
        return EnumValue(
            nominal=value.nominal,
            display_name=value.display_name,
            variant=value.variant,
            fields=dict(value.fields),
        )
    if isinstance(value, ExceptionValue):
        return ExceptionValue(
            nominal=value.nominal, display_name=value.display_name, fields=dict(value.fields)
        )
    return value


def deep_copy_value(value: Value) -> Value:
    """Return a fully independent deep copy of *value*, preserving sharing."""
    return _deep_copy(value, {})


def _memo_copy(
    value: Value,
    memo: dict[int, Value],
    make_shell: Callable[[], _ShellT],
    fill_shell: Callable[[_ShellT], None],
) -> Value:
    """Return the memoized copy of *value*, or build, register, and fill a fresh one.

    ``make_shell`` runs only on a cache miss. For every recursive container
    kind it returns an empty shell that ``fill_shell`` populates once the
    shell is already registered in *memo* — so a self-referential container's
    re-entrant reference resolves to the shell already under construction
    instead of recursing forever. For the atomic ``json`` case ``make_shell``
    performs the whole copy up front and ``fill_shell`` is a no-op.
    """
    cached = memo.get(id(value))
    if cached is not None:
        return cached
    shell = make_shell()
    memo[id(value)] = shell
    fill_shell(shell)
    return shell


def _fill_field_dict(dest: dict[str, Value], src: dict[str, Value], memo: dict[int, Value]) -> None:
    """Populate *dest* with a deep copy of every entry in *src*."""
    for k, v in src.items():
        dest[k] = _deep_copy(v, memo)


def _deep_copy(value: Value, memo: dict[int, Value]) -> Value:
    if isinstance(value, ArrayValue):

        def make_array_shell() -> ArrayValue:
            return ArrayValue(elements=[])

        def fill_array_shell(shell: ArrayValue) -> None:
            for e in value.elements:
                shell.elements.append(_deep_copy(e, memo))

        return _memo_copy(value, memo, make_array_shell, fill_array_shell)
    if isinstance(value, DictValue):
        return _memo_copy(
            value,
            memo,
            lambda: DictValue(entries={}),
            lambda shell: _fill_field_dict(shell.entries, value.entries, memo),
        )
    if isinstance(value, RecordValue):
        return _memo_copy(
            value,
            memo,
            lambda: RecordValue(nominal=value.nominal, display_name=value.display_name, fields={}),
            lambda shell: _fill_field_dict(shell.fields, value.fields, memo),
        )
    if isinstance(value, EnumValue):
        return _memo_copy(
            value,
            memo,
            lambda: EnumValue(
                nominal=value.nominal,
                display_name=value.display_name,
                variant=value.variant,
                fields={},
            ),
            lambda shell: _fill_field_dict(shell.fields, value.fields, memo),
        )
    if isinstance(value, ExceptionValue):
        return _memo_copy(
            value,
            memo,
            lambda: ExceptionValue(
                nominal=value.nominal, display_name=value.display_name, fields={}
            ),
            lambda shell: _fill_field_dict(shell.fields, value.fields, memo),
        )
    if isinstance(value, JsonValue):
        return _memo_copy(
            value,
            memo,
            lambda: JsonValue(_stdlib_copy.deepcopy(value.raw)),
            lambda shell: None,
        )
    return value
