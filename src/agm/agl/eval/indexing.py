"""Pure array/dict index get/set helpers for the AgL evaluator.

Used by the IR evaluator.
This module is the single source of truth for array/dict indexing semantics.

IMPORTANT: Only imports from stdlib, agm.agl.semantics.values, and agm.agl.ir.operations.
No syntax, scope, or typecheck imports are permitted here.
"""

from __future__ import annotations

from typing import assert_never

from agm.agl.ir.operations import IndexKind
from agm.agl.semantics.values import ArrayValue, DictValue, IntValue, TextValue, Value

__all__ = [
    "AglIndexOutOfRange",
    "AglMissingKey",
    "index_get",
    "index_set",
]


class AglIndexOutOfRange(Exception):
    """Sentinel: array index out of range."""

    def __init__(self, index: int, length: int) -> None:
        super().__init__(f"Array index {index} out of range for length {length}")
        self.index = index
        self.length = length


class AglMissingKey(Exception):
    """Sentinel: dict key missing."""

    def __init__(self, key: str) -> None:
        super().__init__(f"Dict key {key!r} is missing")
        self.key = key


def _normalize_array_index(index: int, length: int) -> int:
    """Normalize a (possibly negative) array index; raises AglIndexOutOfRange if OOB."""
    normalized = index if index >= 0 else length + index
    if normalized < 0 or normalized >= length:
        raise AglIndexOutOfRange(index, length)
    return normalized


def index_get(kind: IndexKind, container: Value, index: Value) -> Value:
    """Get a value from an array or dict container by index."""
    match kind:
        case IndexKind.ARRAY:
            if not isinstance(container, ArrayValue):
                raise AssertionError(
                    f"index_get ARRAY: expected ArrayValue, got {type(container).__name__}"
                )
            if not isinstance(index, IntValue):
                raise AssertionError(
                    f"index_get ARRAY: expected IntValue index, got {type(index).__name__}"
                )
            normalized = _normalize_array_index(index.value, len(container.elements))
            return container.elements[normalized]
        case IndexKind.DICT:
            if not isinstance(container, DictValue):
                raise AssertionError(
                    f"index_get DICT: expected DictValue, got {type(container).__name__}"
                )
            if not isinstance(index, TextValue):
                raise AssertionError(
                    f"index_get DICT: expected TextValue index, got {type(index).__name__}"
                )
            if index.value not in container.entries:
                raise AglMissingKey(index.value)
            return container.entries[index.value]
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def index_set(kind: IndexKind, container: Value, index: Value, value: Value) -> Value:
    """Return a new immutable container with the slot at index replaced by value."""
    match kind:
        case IndexKind.ARRAY:
            if not isinstance(container, ArrayValue):
                raise AssertionError(
                    f"index_set ARRAY: expected ArrayValue, got {type(container).__name__}"
                )
            if not isinstance(index, IntValue):
                raise AssertionError(
                    f"index_set ARRAY: expected IntValue index, got {type(index).__name__}"
                )
            normalized = _normalize_array_index(index.value, len(container.elements))
            elements = list(container.elements)
            elements[normalized] = value
            return ArrayValue(tuple(elements))
        case IndexKind.DICT:
            if not isinstance(container, DictValue):
                raise AssertionError(
                    f"index_set DICT: expected DictValue, got {type(container).__name__}"
                )
            if not isinstance(index, TextValue):
                raise AssertionError(
                    f"index_set DICT: expected TextValue index, got {type(index).__name__}"
                )
            entries = dict(container.entries)
            entries[index.value] = value
            return DictValue(entries)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)
