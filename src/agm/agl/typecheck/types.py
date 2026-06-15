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
    with name ``"Exception"``, only ``message``/``trace_id`` fields, and
    ``abstract=True`` (it is catchable as the hierarchy root but not
    constructible).
    """

    name: str
    fields: Mapping[str, Type] = field(default_factory=dict)
    abstract: bool = False

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


def is_json_shaped(value_type: Type) -> bool:
    """Return ``True`` if ``value_type`` is JSON-shaped (design §5.8 rule 3).

    JSON-shaped types are the values that may inhabit a ``json`` slot:
    ``null``/``json``, ``bool``, ``int``, ``decimal``, ``text``, and
    ``list``/``dict`` whose element/value types are themselves JSON-shaped.
    Records, enums, and exceptions are **not** JSON-shaped — to embed one in a
    ``json`` value they must be rendered explicitly (e.g. ``${review as json}``).
    """
    if isinstance(value_type, (TextType, JsonType, BoolType, IntType, DecimalType)):
        return True
    if isinstance(value_type, ListType):
        return is_json_shaped(value_type.elem)
    if isinstance(value_type, DictType):
        return is_json_shaped(value_type.value)
    # RecordType, EnumType, ExceptionType are not JSON-shaped.
    return False


def comparable_types(left: Type, right: Type) -> bool:
    """Return ``True`` if ``left`` and ``right`` may be compared (design §5.8 r4).

    Equality (``=``, ``!=``) and ordering comparisons require both operands to
    have the **same** type after the single ``int → decimal`` widening.  Unlike
    :func:`is_assignable`, ``json`` does **not** absorb JSON-shaped scalars here:
    ``json = json`` is allowed but ``json`` vs any non-``json`` type is a static
    error (rule 4 as written).  Records/enums/exceptions compare only with their
    own exact type.
    """
    if left == right:
        return True
    # The only cross-type comparison is numeric int↔decimal (either direction).
    numeric = (IntType, DecimalType)
    return isinstance(left, numeric) and isinstance(right, numeric)


def is_assignable(value_type: Type, target_type: Type) -> bool:
    """Return ``True`` if ``value_type`` is assignable to ``target_type``.

    Implicit coercions (design §5.8):

    1. ``int → decimal`` widening is the only scalar coercion.
    2. ``json`` accepts any JSON-shaped value (rule 3): ``null``/``json``,
       ``bool``, ``int``, ``decimal``, ``text``, and ``list``/``dict`` of
       JSON-shaped types.  Records/enums/exceptions are rejected.

    All other assignments require exact structural equality.
    """
    if value_type == target_type:
        return True
    # Single scalar coercion: int can widen to decimal.
    if isinstance(value_type, IntType) and isinstance(target_type, DecimalType):
        return True
    # json accepts any JSON-shaped value (records/enums/exceptions excluded).
    if isinstance(target_type, JsonType):
        return is_json_shaped(value_type)
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
    abstract=True,
)

# Concrete built-in exceptions — exact §8.1 table.
# Every exception includes the base fields (message, trace_id) plus the
# additional fields listed in the design's "Additional fields" section.
#
# Changes from prior draft vs §8.1:
#   - AgentCallError: added agent: text and metadata: json (§8.1 §0 resolution 11)
#   - UndefinedVariableError: added with name: text (§8.1)
#   - ImmutableBindingError: added with name: text, operation: text (§8.1)
#   - ValidationError: REMOVED — §8.1 does not list it as a catchable exception;
#     agm.agl.runtime.request.ValidationError is a Python-level record shape
#     embedded in AgentParseError.validation_errors (design §7.5), not an AgL type.
BUILTIN_EXCEPTIONS: dict[str, ExceptionType] = {
    "Exception": EXCEPTION_BASE,
    # §8.1 AgentCallError: agent/cause/metadata (§0 resolution 11: cause is
    # enumerated "spawn_failure"|"nonzero_exit"|"timeout"; metadata carries
    # exit code, stderr tail, elapsed — all stored in the json field).
    "AgentCallError": ExceptionType(
        name="AgentCallError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "agent": TextType(),
            "cause": TextType(),
            "metadata": JsonType(),
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
    "MaxIterationsExceeded": ExceptionType(
        name="MaxIterationsExceeded",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "limit": IntType(),
            "condition": TextType(),
            "last_condition_value": BoolType(),
            "metadata": JsonType(),
        },
    ),
    "MatchError": ExceptionType(
        name="MatchError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "scrutinee_type": TextType(),
            "scrutinee": JsonType(),
        },
    ),
    "TypeError": ExceptionType(
        name="TypeError",
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
    # §8.1: statically prevented in v1 (scope/typecheck reject set on immutable
    # bindings and undeclared names), but still listed as catchable runtime
    # exceptions for any runtime paths that bypass the static passes.
    "UndefinedVariableError": ExceptionType(
        name="UndefinedVariableError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "name": TextType(),
        },
    ),
    "ImmutableBindingError": ExceptionType(
        name="ImmutableBindingError",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
            "name": TextType(),
            "operation": TextType(),
        },
    ),
    "Abort": ExceptionType(
        name="Abort",
        fields={
            "message": TextType(),
            "trace_id": TextType(),
        },
    ),
}

# Names of built-in exception types (cannot be redeclared as records/enums/aliases).
BUILTIN_EXCEPTION_NAMES: frozenset[str] = frozenset(BUILTIN_EXCEPTIONS)
