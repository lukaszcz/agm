"""Pure array/dict/text index get/set helpers for the AgL evaluator.

Used by the IR evaluator.
This module is the single source of truth for array/dict/text indexing semantics.

IMPORTANT: Only imports from stdlib, agm.agl.semantics.values, and agm.agl.ir.operations.
No syntax, scope, or typecheck imports are permitted here.
"""

from __future__ import annotations

from typing import assert_never, cast

from agm.agl.ir.operations import IndexKind, MutableIndexKind
from agm.agl.semantics.values import ArrayValue, DictValue, IntValue, TextValue, Value

__all__ = [
    "AglIndexOutOfRange",
    "AglMissingKey",
    "index_get",
    "index_set",
]


class AglIndexOutOfRange(Exception):
    """Sentinel: array or text index is out of range."""

    def __init__(self, index: int, length: int, kind: IndexKind) -> None:
        label = "Array" if kind is IndexKind.ARRAY else "Text"
        super().__init__(f"{label} index {index} out of range for length {length}")
        self.index = index
        self.length = length


class AglMissingKey(Exception):
    """Sentinel: dict key missing. *key* is the key ``Value``, not its rendering."""

    def __init__(self, key: Value) -> None:
        super().__init__("dict key is missing")
        self.key = key


def _normalize_index(index: int, length: int, kind: IndexKind) -> int:
    """Normalize an array or text index; raise AglIndexOutOfRange if it is out of range."""
    normalized = index if index >= 0 else length + index
    if normalized < 0 or normalized >= length:
        raise AglIndexOutOfRange(index, length, kind)
    return normalized


def index_get(kind: IndexKind, container: Value, index: Value) -> Value:
    """Get a value from an array, dict, or text container by index."""
    match kind:
        case IndexKind.ARRAY:
            container = cast(ArrayValue, container)
            index = cast(IntValue, index)
            normalized = _normalize_index(index.value, len(container.elements), IndexKind.ARRAY)
            return container.elements[normalized]
        case IndexKind.TEXT:
            container = cast(TextValue, container)
            index = cast(IntValue, index)
            normalized = _normalize_index(index.value, len(container.value), IndexKind.TEXT)
            return TextValue(container.value[normalized])
        case IndexKind.DICT:
            container = cast(DictValue, container)
            found = container.lookup(index)
            if found is None:
                raise AglMissingKey(index)
            return found
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def index_set(kind: MutableIndexKind, container: Value, index: Value, value: Value) -> None:
    """Mutate an array or dict *container* in place, storing *value* at *index*.

    An out-of-range array index raises ``AglIndexOutOfRange``. A dict
    assignment updates an **existing key only**; a missing key raises
    ``AglMissingKey`` rather than inserting it, so this is the single source
    of truth for the missing-key rule (callers must not pre-check via
    ``index_get``). Text is immutable and never reaches this helper — its
    ``kind`` excludes ``IndexKind.TEXT`` at the type level.
    """
    match kind:
        case IndexKind.ARRAY:
            container = cast(ArrayValue, container)
            index = cast(IntValue, index)
            normalized = _normalize_index(index.value, len(container.elements), IndexKind.ARRAY)
            container.elements[normalized] = value
        case IndexKind.DICT:
            container = cast(DictValue, container)
            if not container.update_existing(index, value):
                raise AglMissingKey(index)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)
