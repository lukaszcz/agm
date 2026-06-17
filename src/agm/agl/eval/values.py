"""AgL runtime value hierarchy (Component 6).

The closed ``Value`` union represents every value that can appear in a
running AgL program.  All members are frozen dataclasses so they can be
stored in dicts and compared by equality.

M1 types implemented:
  TextValue, IntValue, DecimalValue, BoolValue, JsonValue

Full-hierarchy types (additive additions in M2+):
  ListValue, DictValue, RecordValue, EnumValue, ExceptionValue

``JsonValue`` wraps an ``object``-typed JSON tree — the single dynamic
boundary (mirroring ``config/sandbox/srt.py``).  No other value type uses
``Any`` or ``object`` for its payload.
"""

from __future__ import annotations

import decimal
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agm.agl.eval.scope import Scope
    from agm.agl.syntax.nodes import Expr
    from agm.agl.typecheck.types import Type

# ---------------------------------------------------------------------------
# JSON-tree comparison helpers (used by JsonValue.__eq__)
# ---------------------------------------------------------------------------


def _json_eq(left: object, right: object) -> bool:
    """Compare two JSON-shaped trees with bool-guarded numeric equivalence.

    Mirrors the semantics in ``interpreter._json_eq``: JSON numbers compare
    numerically (``1 == 1.0``), but ``bool`` is a distinct JSON kind and never
    compares equal to a number (no Python ``True == 1`` conflation).  Containers
    recurse structurally; ``text`` and ``null`` compare exactly.

    Defined here so ``JsonValue.__eq__`` can use it without importing from
    interpreter (which would create a circular dependency).
    """
    # bool first: Python treats bool as a subclass of int, so ``True == 1``.
    # Guard: a bool only equals another bool of the same value.
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, decimal.Decimal)) and isinstance(
        right, (int, decimal.Decimal)
    ):
        return decimal.Decimal(left) == decimal.Decimal(right)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_json_eq(left[i], right[i]) for i in range(len(left)))
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        return all(_json_eq(left[k], right[k]) for k in left)
    return left == right


def _json_hash(obj: object) -> int:
    """Stable hash for a JSON-shaped tree.

    Must be consistent with ``_json_eq``: objects that compare equal must hash
    equal.  Because ``_json_eq`` treats numeric int/Decimal equivalently, we
    normalise numbers to ``Decimal`` before hashing.  Lists and dicts recurse;
    bools are guarded so ``True`` never hashes the same as ``1``.
    """
    if isinstance(obj, bool):
        # Hash True/False distinctly from integers.
        return hash(("__bool__", obj))
    if isinstance(obj, (int, decimal.Decimal)):
        # Normalise to Decimal so 1 and Decimal("1") hash the same.
        return hash(decimal.Decimal(obj))
    if isinstance(obj, list):
        return hash(tuple(_json_hash(e) for e in obj))
    if isinstance(obj, dict):
        return hash(frozenset((_json_hash(k), _json_hash(v)) for k, v in obj.items()))
    return hash(obj)


# ---------------------------------------------------------------------------
# Primitive value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextValue:
    """A ``text`` value: a plain Python ``str``."""

    value: str


@dataclass(frozen=True, slots=True)
class IntValue:
    """An ``int`` value: an arbitrary-precision Python ``int``."""

    value: int


@dataclass(frozen=True, slots=True)
class DecimalValue:
    """A ``decimal`` value: an exact ``decimal.Decimal``."""

    value: decimal.Decimal


@dataclass(frozen=True, slots=True)
class BoolValue:
    """A ``bool`` value."""

    value: bool


@dataclass(frozen=True, slots=True, eq=False)
class JsonValue:
    """A ``json`` value: any JSON-shaped Python object (the dynamic boundary).

    The ``raw`` field is ``object`` to allow ``None``, dicts, lists, strings,
    ints, floats, and bools — exactly what ``json.loads`` can return.  All
    operations on the payload use ``isinstance`` guards, never bare ``Any``
    access.

    ``__eq__`` delegates to ``_json_eq`` so that JSON bool/number conflation is
    prevented inside containers (e.g. ``JsonValue([True]) != JsonValue([1])``),
    consistent with the top-level ``json = json`` comparison semantics
    (design §5.8/§11.9).  ``__hash__`` is consistent with ``__eq__`` via
    ``_json_hash``.
    """

    raw: object

    def __eq__(self, other: object) -> bool:
        if isinstance(other, JsonValue):
            return _json_eq(self.raw, other.raw)
        return NotImplemented

    def __hash__(self) -> int:
        return _json_hash(self.raw)


# ---------------------------------------------------------------------------
# Container value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ListValue:
    """A ``list[T]`` value: an immutable tuple of ``Value`` items."""

    elements: tuple[Value, ...]


@dataclass(frozen=True, slots=True)
class DictValue:
    """A ``dict[text, V]`` value: an immutable mapping of str → Value."""

    # Stored as a plain dict; frozen by convention (no mutation after creation).
    entries: dict[str, Value] = field(default_factory=dict)

    def __hash__(self) -> int:
        # Hash via hash(v) so that the contract hash(a) == hash(b) whenever a == b
        # is preserved.  JsonValue.__hash__ uses _json_hash (order-insensitive,
        # numeric-canonical), so equal-but-differently-ordered or int-vs-Decimal
        # payloads hash the same.
        return hash(tuple(sorted((k, hash(v)) for k, v in self.entries.items())))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DictValue):
            return self.entries == other.entries
        return NotImplemented


@dataclass(frozen=True, slots=True)
class RecordValue:
    """A record-typed value."""

    type_name: str
    fields: dict[str, Value] = field(default_factory=dict)

    def __hash__(self) -> int:
        # Use hash(v) rather than repr(v) so that the eq/hash contract holds:
        # equal values (e.g. JsonValue(1) == JsonValue(Decimal("1.0"))) hash the same.
        return hash(
            (self.type_name, tuple(sorted((k, hash(v)) for k, v in self.fields.items())))
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RecordValue):
            return self.type_name == other.type_name and self.fields == other.fields
        return NotImplemented


@dataclass(frozen=True, slots=True)
class EnumValue:
    """An enum-typed value: the active variant name plus any payload fields."""

    type_name: str
    variant: str
    fields: dict[str, Value] = field(default_factory=dict)

    def __hash__(self) -> int:
        # Use hash(v) rather than repr(v) so that the eq/hash contract holds.
        return hash(
            (
                self.type_name,
                self.variant,
                tuple(sorted((k, hash(v)) for k, v in self.fields.items())),
            )
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EnumValue):
            return (
                self.type_name == other.type_name
                and self.variant == other.variant
                and self.fields == other.fields
            )
        return NotImplemented


@dataclass(frozen=True, slots=True)
class ExceptionValue:
    """A built-in AgL exception value.

    ``type_name`` is the exception class name (e.g. ``"AgentParseError"``).
    ``fields`` maps the exception's declared field names to their values.
    The ``"message"`` and ``"trace_id"`` fields are always present (base
    ``Exception`` contract).
    """

    type_name: str
    fields: dict[str, Value] = field(default_factory=dict)

    def __hash__(self) -> int:
        # Use hash(v) rather than repr(v) so that the eq/hash contract holds.
        return hash(
            (self.type_name, tuple(sorted((k, hash(v)) for k, v in self.fields.items())))
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ExceptionValue):
            return self.type_name == other.type_name and self.fields == other.fields
        return NotImplemented


# ---------------------------------------------------------------------------
# v2 value types: unit, agent handle, and closure
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UnitValue:
    """The single value of the ``unit`` type: ``()``."""


UNIT_VALUE: UnitValue = UnitValue()


@dataclass(frozen=True, slots=True)
class AgentValue:
    """A first-class agent handle — opaque; not renderable or comparable."""

    name: str


@dataclass(slots=True)
class Closure:
    """A first-class function value — a lambda or def closure.

    ``env`` is the scope captured at closure creation time.
    ``params`` is an ordered tuple of (name, default_expr_or_None) pairs.
    ``body`` is the unevaluated body expression.
    ``return_type`` is the declared return type (used for coercion).
    """

    env: "Scope"
    params: "tuple[tuple[str, Expr | None], ...]"
    body: "Expr"
    return_type: "Type"

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


# ---------------------------------------------------------------------------
# Closed Value union
# ---------------------------------------------------------------------------

Value = (
    TextValue
    | IntValue
    | DecimalValue
    | BoolValue
    | JsonValue
    | ListValue
    | DictValue
    | RecordValue
    | EnumValue
    | ExceptionValue
    | UnitValue
    | AgentValue
    | Closure
)
