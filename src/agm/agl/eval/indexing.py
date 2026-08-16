"""Pure array/dict/text index get/set helpers for the AgL evaluator.

Used by the IR evaluator.
This module is the single source of truth for array/dict/text indexing semantics.

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
    """Sentinel: array or text index is out of range."""

    def __init__(self, index: int, length: int, kind: IndexKind) -> None:
        label = "Array" if kind is IndexKind.ARRAY else "Text"
        super().__init__(f"{label} index {index} out of range for length {length}")
        self.index = index
        self.length = length


class AglMissingKey(Exception):
    """Sentinel: dict key missing."""

    def __init__(self, key: str) -> None:
        super().__init__(f"Dict key {key!r} is missing")
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
            if not isinstance(container, ArrayValue):
                raise AssertionError(
                    f"index_get ARRAY: expected ArrayValue, got {type(container).__name__}"
                )
            if not isinstance(index, IntValue):
                raise AssertionError(
                    f"index_get ARRAY: expected IntValue index, got {type(index).__name__}"
                )
            normalized = _normalize_index(index.value, len(container.elements), IndexKind.ARRAY)
            return container.elements[normalized]
        case IndexKind.TEXT:
            if not isinstance(container, TextValue):
                raise AssertionError(
                    f"index_get TEXT: expected TextValue, got {type(container).__name__}"
                )
            if not isinstance(index, IntValue):
                raise AssertionError(
                    f"index_get TEXT: expected IntValue index, got {type(index).__name__}"
                )
            normalized = _normalize_index(index.value, len(container.value), IndexKind.TEXT)
            return TextValue(container.value[normalized])
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


def index_set(kind: IndexKind, container: Value, index: Value, value: Value) -> None:
    """Mutate an array or dict *container* in place, storing *value* at *index*.

    An out-of-range array index raises ``AglIndexOutOfRange``; text is
    immutable and cannot be assigned through this helper. A dict assignment
    updates an **existing key only**; a missing key raises
    ``AglMissingKey`` rather than inserting it, so this is the single source
    of truth for the missing-key rule (callers must not pre-check via
    ``index_get``).
    """
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
            normalized = _normalize_index(index.value, len(container.elements), IndexKind.ARRAY)
            container.elements[normalized] = value
        case IndexKind.TEXT:
            raise AssertionError("index_set TEXT: text is immutable")
        case IndexKind.DICT:
            if not isinstance(container, DictValue):
                raise AssertionError(
                    f"index_set DICT: expected DictValue, got {type(container).__name__}"
                )
            if not isinstance(index, TextValue):
                raise AssertionError(
                    f"index_set DICT: expected TextValue index, got {type(index).__name__}"
                )
            if index.value not in container.entries:
                raise AglMissingKey(index.value)
            container.entries[index.value] = value
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)
