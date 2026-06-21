"""Closed operation enumerations and coercion descriptors for the AgL IR.

Enums are derived directly from the operator sets supported by AgL:
- ``ArithOp``: binary arithmetic operators (``+``, ``-``, ``*``, ``/``).
  Derived from ``BinOp.ADD/SUB/MUL/DIV`` in ``agm.agl.syntax.nodes``.
  Note: there is no modulo operator in AgL v2 (the ``BinOp`` enum has no MOD).
- ``CmpOp``: comparison operators (``=``, ``!=``, ``<``, ``<=``, ``>``, ``>=``).
  Derived from ``BinOp.EQ/NEQ/LT/LE/GT/GE`` in ``agm.agl.syntax.nodes``.
  The ``in`` operator is lowered to ``IrContains`` (not a CmpOp).
  ``and``/``or`` are lowered to ``IrAnd``/``IrOr`` (deferred to M3).

``Coercion`` is a closed union of frozen dataclasses.  An identity coercion
(no-op) is represented by ``None`` at use sites — it is not a member here.
Container coercions carry only child ops that do real work.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

__all__ = [
    "ArithKind",
    "ArithOp",
    "CmpOp",
    "Coercion",
    "CompareKind",
    "ContainsKind",
    "IntToDecimal",
    "MapDictValues",
    "MapEnumFields",
    "MapList",
    "MapRecordFields",
    "NumericKind",
    "ToJson",
    "UnaryOp",
]


# ---------------------------------------------------------------------------
# Operation enums (D3)
# ---------------------------------------------------------------------------


class ArithOp(enum.Enum):
    """Closed set of binary arithmetic operators in AgL.

    Derived from the arithmetic branch of ``BinOp`` in
    ``agm.agl.syntax.nodes``: ADD(+), SUB(-), MUL(*), DIV(/).
    AgL v2 has no modulo operator.
    """

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"


class ArithKind(enum.Enum):
    """Kind tag for arithmetic operations: integer, decimal, or text (ADD only)."""

    INT = "int"
    DECIMAL = "decimal"
    TEXT = "text"


class CmpOp(enum.Enum):
    """Closed set of comparison operators in AgL.

    Derived from the equality/ordering branches of ``BinOp`` in
    ``agm.agl.syntax.nodes``: EQ(=), NEQ(!=), LT(<), LE(<=), GT(>), GE(>=).
    The ``in`` operator is a separate ``IrContains`` node (see M3).
    """

    EQ = "="
    NEQ = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="


class NumericKind(enum.Enum):
    """Kind tag for numeric operations: integer or decimal."""

    INT = "int"
    DECIMAL = "decimal"


class CompareKind(enum.Enum):
    """Kind tag for comparison operations: integer, decimal, text, or structural."""

    INT = "int"
    DECIMAL = "decimal"
    TEXT = "text"
    STRUCTURAL = "structural"


class ContainsKind(enum.Enum):
    """Kind tag for the ``in`` containment operator: list, dict, or text."""

    LIST = "list"
    DICT = "dict"
    TEXT = "text"


class UnaryOp(enum.Enum):
    """Kind tag for unary operations: NOT (logical negation) or NEG (numeric negation)."""

    NOT = "not"
    NEG = "neg"


# ---------------------------------------------------------------------------
# Coercion closed union (D3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IntToDecimal:
    """Coercion: widen an ``int`` value to ``decimal``."""


@dataclass(frozen=True, slots=True)
class ToJson:
    """Coercion: convert a structured value to its JSON representation (text)."""


@dataclass(frozen=True, slots=True)
class MapList:
    """Coercion: apply a child coercion to every element of a list."""

    item: "Coercion"


@dataclass(frozen=True, slots=True)
class MapDictValues:
    """Coercion: apply a child coercion to every value of a dict (keys unchanged)."""

    value: "Coercion"


@dataclass(frozen=True, slots=True)
class MapRecordFields:
    """Coercion: apply per-field coercions to a record value.

    ``fields`` is a tuple of ``(field_name, child_coercion)`` pairs.
    The target nominal is known from the enclosing construct; field coercions
    are addressed by name only.
    """

    fields: tuple[tuple[str, "Coercion"], ...]


@dataclass(frozen=True, slots=True)
class MapEnumFields:
    """Coercion: apply per-variant/per-field coercions to an enum value.

    ``variants`` is a tuple of ``(variant_name, field_coercions)`` pairs where
    ``field_coercions`` is a tuple of ``(field_name, child_coercion)`` pairs.
    The target nominal is known from the enclosing construct.
    """

    variants: tuple[tuple[str, tuple[tuple[str, "Coercion"], ...]], ...]


#: Closed union of coercion operations.  An identity (no-op) coercion is
#: represented by ``None`` at use sites; it is NOT a member of this union.
Coercion = IntToDecimal | ToJson | MapList | MapDictValues | MapRecordFields | MapEnumFields
