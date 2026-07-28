"""Closed operation enumerations and coercion descriptors for the AgL IR.

Enums are derived directly from the operator sets supported by AgL:
- ``ArithOp``: binary arithmetic operators (``+``, ``-``, ``*``, ``/``).
  Derived from ``BinOp.ADD/SUB/MUL/DIV`` in ``agm.agl.syntax.nodes``.
  Note: there is no modulo operator in AgL (the ``BinOp`` enum has no MOD).
- ``CmpOp``: comparison operators (``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``).
  Derived from ``BinOp.EQ/NEQ/LT/LE/GT/GE`` in ``agm.agl.syntax.nodes``.
  The ``in`` operator is lowered to ``IrContains`` (not a CmpOp).
  ``and``/``or`` are lowered to ``IrAnd``/``IrOr``.

``Coercion`` is a closed union of frozen dataclasses.  An identity coercion
(no-op) is represented by ``None`` at use sites — it is not a member here.
Every coercion is a scalar leaf conversion; none rebuilds a structure.
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
    "CopyKind",
    "IndexKind",
    "IntToDecimal",
    "IterKind",
    "NumericKind",
    "ToJson",
    "UnaryOp",
]


# ---------------------------------------------------------------------------
# Operation enums
# ---------------------------------------------------------------------------


class ArithOp(enum.Enum):
    """Closed set of binary arithmetic operators in AgL.

    Derived from the arithmetic branch of ``BinOp`` in
    ``agm.agl.syntax.nodes``: ADD(+), SUB(-), MUL(*), DIV(/).
    AgL has no modulo operator.
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
    ``agm.agl.syntax.nodes``: EQ(==), NEQ(!=), LT(<), LE(<=), GT(>), GE(>=).
    The ``in`` operator is a separate ``IrContains`` node.
    """

    EQ = "=="
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
    """Kind tag for the ``in`` containment operator: array, dict, or text."""

    ARRAY = "array"
    DICT = "dict"
    TEXT = "text"


class IndexKind(enum.Enum):
    """Kind tag for index access: array or dict."""

    ARRAY = "array"
    DICT = "dict"


class CopyKind(enum.Enum):
    """Kind tag for value copying: deep or shallow."""

    DEEP = "deep"
    SHALLOW = "shallow"


class UnaryOp(enum.Enum):
    """Kind tag for unary operations: NOT (logical negation) or NEG (numeric negation)."""

    NOT = "not"
    NEG = "neg"


class IterKind(enum.Enum):
    """Kind tag for for-loop iteration: array elements, dict keys, or text chars."""

    ARRAY = "array"
    DICT_KEYS = "dict_keys"
    TEXT = "text"


# ---------------------------------------------------------------------------
# Coercion closed union
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IntToDecimal:
    """Coercion: widen an ``int`` value to ``decimal``."""


@dataclass(frozen=True, slots=True)
class ToJson:
    """Coercion: convert a scalar value to its JSON representation."""


#: Closed union of coercion operations.  An identity (no-op) coercion is
#: represented by ``None`` at use sites; it is NOT a member of this union.
Coercion = IntToDecimal | ToJson
