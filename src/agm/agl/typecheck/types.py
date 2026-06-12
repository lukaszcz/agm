"""Semantic type model for the AgL type checker.

These are *resolved* nominal types distinct from the syntactic ``TypeExpr``
hierarchy in ``agm.agl.syntax.types``.  Aliases are resolved transparently
to their target — ``TypeAlias`` nodes never appear here.

Type hierarchy
--------------
- ``TextType`` — the ``text`` primitive.
- ``JsonType`` — the ``json`` primitive (any JSON-shaped value).
- ``BoolType`` — the ``bool`` primitive.
- ``IntType`` — the ``int`` primitive (arbitrary-precision integer).
- ``DecimalType`` — the ``decimal`` primitive (exact fixed-point).
- ``ListType(elem)`` — ``list[T]``.
- ``DictType(value)`` — ``dict[text, V]`` (keys are always ``text`` in v1).
- ``RecordType(name, fields)`` — a ``record`` nominal type.
- ``EnumType(name, variants)`` — an ``enum`` nominal type.
- ``ExceptionType(name, fields)`` — a built-in exception type.

``Type`` is the closed union of all semantic types.

Single coercion rule (design §5.8)
-------------------------------------
``int → decimal`` widening is the **only** implicit type coercion.  Use
:func:`is_assignable` to check assignability with this single coercion
applied.

Type-kind strings (for codec capability lookup)
------------------------------------------------
Each ``Type`` exposes a ``kind`` property — a lower-cased string identifying
the type's kind in the ``HostCapabilities.codec_kinds`` maps.  E.g.
``TextType().kind == "text"``, ``RecordType(...).kind == "record"``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Primitive types (singletons-by-construction; frozen dataclasses)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextType:
    """The ``text`` built-in type."""

    @property
    def kind(self) -> str:
        return "text"

    def __repr__(self) -> str:
        return "text"


@dataclass(frozen=True, slots=True)
class JsonType:
    """The ``json`` built-in type (any JSON value)."""

    @property
    def kind(self) -> str:
        return "json"

    def __repr__(self) -> str:
        return "json"


@dataclass(frozen=True, slots=True)
class BoolType:
    """The ``bool`` built-in type."""

    @property
    def kind(self) -> str:
        return "bool"

    def __repr__(self) -> str:
        return "bool"


@dataclass(frozen=True, slots=True)
class IntType:
    """The ``int`` built-in type."""

    @property
    def kind(self) -> str:
        return "int"

    def __repr__(self) -> str:
        return "int"


@dataclass(frozen=True, slots=True)
class DecimalType:
    """The ``decimal`` built-in type (exact fixed-point)."""

    @property
    def kind(self) -> str:
        return "decimal"

    def __repr__(self) -> str:
        return "decimal"


# ---------------------------------------------------------------------------
# Parameterised container types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ListType:
    """``list[T]`` — a homogeneous list."""

    elem: Type

    @property
    def kind(self) -> str:
        return "list"

    def __repr__(self) -> str:
        return f"list[{self.elem!r}]"


@dataclass(frozen=True, slots=True)
class DictType:
    """``dict[text, V]`` — string-keyed dict."""

    value: Type

    @property
    def kind(self) -> str:
        return "dict"

    def __repr__(self) -> str:
        return f"dict[text, {self.value!r}]"


# ---------------------------------------------------------------------------
# Nominal types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecordType:
    """A ``record`` nominal type.

    ``fields`` maps field name → field type.
    """

    name: str
    fields: Mapping[str, Type]

    @property
    def kind(self) -> str:
        return "record"

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class EnumType:
    """An ``enum`` nominal type.

    ``variants`` maps variant name → mapping of field names → field types.
    """

    name: str
    variants: Mapping[str, Mapping[str, Type]]

    @property
    def kind(self) -> str:
        return "enum"

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class ExceptionType:
    """A built-in exception type.

    ``fields`` maps field name → field type.
    The abstract ``Exception`` base is represented as an ``ExceptionType``
    with name ``"Exception"`` and only ``message``/``trace_id`` fields.
    """

    name: str
    fields: Mapping[str, Type] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "exception"

    def __repr__(self) -> str:
        return self.name


# Closed union of all semantic types.
Type = (
    TextType
    | JsonType
    | BoolType
    | IntType
    | DecimalType
    | ListType
    | DictType
    | RecordType
    | EnumType
    | ExceptionType
)


# ---------------------------------------------------------------------------
# Assignability helper (single coercion: int → decimal)
# ---------------------------------------------------------------------------


def is_assignable(value_type: Type, target_type: Type) -> bool:
    """Return ``True`` if ``value_type`` is assignable to ``target_type``.

    The only implicit coercion is ``int → decimal`` widening (design §5.8).
    All other assignments require exact structural equality.
    """
    if value_type == target_type:
        return True
    # Single coercion: int literal (or int-typed expression) can widen to decimal.
    if isinstance(value_type, IntType) and isinstance(target_type, DecimalType):
        return True
    return False


# ---------------------------------------------------------------------------
# Built-in exception types (design §8.1)
# ---------------------------------------------------------------------------

# Abstract base: only message + trace_id fields.
EXCEPTION_BASE = ExceptionType(
    name="Exception",
    fields={
        "message": TextType(),
        "trace_id": TextType(),
    },
)

# Concrete built-in exceptions (M1 subset; full schemas land in M4).
BUILTIN_EXCEPTIONS: dict[str, ExceptionType] = {
    "Exception": EXCEPTION_BASE,
    "AgentCallError": ExceptionType(
        name="AgentCallError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "cause": TextType(),
        },
    ),
    "AgentParseError": ExceptionType(
        name="AgentParseError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "agent": TextType(),
            "target_type": TextType(),
            "expected_schema": JsonType(),
            "raw": TextType(),
            "normalized_raw": TextType(),
            "validation_errors": JsonType(),
            "attempts": IntType(),
            "metadata": JsonType(),
        },
    ),
    "MaxIterationsExceeded": ExceptionType(
        name="MaxIterationsExceeded",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "limit": IntType(),
        },
    ),
    "MatchError": ExceptionType(
        name="MatchError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
        },
    ),
    "ArithmeticError": ExceptionType(
        name="ArithmeticError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "operation": TextType(),
        },
    ),
    "ExecError": ExceptionType(
        name="ExecError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "command": TextType(),
            "exit_code": IntType(),
            "stdout": TextType(),
            "stderr": TextType(),
            "timed_out": BoolType(),
        },
    ),
    "Abort": ExceptionType(
        name="Abort",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
        },
    ),
    "ValidationError": ExceptionType(
        name="ValidationError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
        },
    ),
    "TypeError": ExceptionType(
        name="TypeError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
        },
    ),
}

# Names of built-in exception types (cannot be redeclared as records/enums/aliases).
BUILTIN_EXCEPTION_NAMES: frozenset[str] = frozenset(BUILTIN_EXCEPTIONS)
