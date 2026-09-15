"""AST nodes for AgL value syntax: a data-only subset of AgL expressions.

Every node carries ``start``/``end`` source offsets. A :class:`CtorNode`'s
``args`` distinguishes the bare form (``None``), an empty call (``()``), and a
call with arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class IntNode:
    """An integer literal."""

    value: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class DecimalNode:
    """A decimal literal (sign folded into ``value``)."""

    value: Decimal
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class BoolNode:
    """A ``true``/``false`` literal."""

    value: bool
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class NullNode:
    """A ``null`` literal."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class TextNode:
    """A decoded text literal."""

    value: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class ArrayNode:
    """An ``[...]`` array literal."""

    items: tuple["ValueNode", ...]
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class DictEntry:
    """One ``key: value`` entry of a dict literal, in source order."""

    key: str
    value: "ValueNode"
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class DictNode:
    """A ``{...}`` dict literal; duplicate keys are kept, not deduped."""

    entries: tuple[DictEntry, ...]
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class ValueArg:
    """One constructor call argument; ``name`` is ``None`` for a positional one."""

    name: str | None
    value: "ValueNode"
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class CtorNode:
    """A constructor reference or call.

    ``args is None`` is the bare form (``Circle``); ``()`` is an empty call
    (``Circle()``); a non-empty tuple carries the call's arguments.
    """

    qualifier: str | None
    name: str
    args: tuple[ValueArg, ...] | None
    start: int
    end: int


type ValueNode = (
    IntNode | DecimalNode | BoolNode | NullNode | TextNode | ArrayNode | DictNode | CtorNode
)
