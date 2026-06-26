"""Single value-model home for the AgL semantics layer.

This module is the **single source of truth** for every runtime value type in
the AgL execution pipeline: leaf primitive value tags, container and nominal
types, IR closures, and the per-invocation frame model (``Cell``, ``Slot``,
``Frame``).

There is exactly one ``Value`` union — the broad 14-member union covering all
leaf primitive, container, nominal, and callable value kinds.

Design constraints
------------------
- Imports ONLY from the Python standard library and ``agm.agl.ir.ids``.
  Must NOT import from ``eval``, ``syntax``, ``scope``, ``typecheck``,
  ``runtime``, or ``modules`` — not even under ``TYPE_CHECKING``.
- All value types are frozen dataclasses with ``__slots__`` for memory
  efficiency.
- ``Cell`` is the sole mutable type (not frozen) — it is a mutable box for
  ``var`` bindings.
"""

from __future__ import annotations

import decimal
from dataclasses import dataclass, field
from typing import TypeAlias

from agm.agl.ir.ids import FunctionId, NominalId, SymbolId

# ---------------------------------------------------------------------------
# JSON-tree comparison helpers
# ---------------------------------------------------------------------------


def _json_eq(left: object, right: object) -> bool:
    """Compare two JSON-shaped trees with bool-guarded numeric equivalence.

    Mirrors the semantics in the interpreter: JSON numbers compare numerically
    (``1 == 1.0``), but ``bool`` is a distinct JSON kind and never compares
    equal to a number (no Python ``True == 1`` conflation).  Containers recurse
    structurally; ``text`` and ``null`` compare exactly.
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
    consistent with the top-level ``json = json`` comparison semantics.
    ``__hash__`` is consistent with ``__eq__`` via ``_json_hash``.
    """

    raw: object

    def __eq__(self, other: object) -> bool:
        if isinstance(other, JsonValue):
            return _json_eq(self.raw, other.raw)
        return NotImplemented

    def __hash__(self) -> int:
        return _json_hash(self.raw)


# ---------------------------------------------------------------------------
# Unit and agent handle value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UnitValue:
    """The single value of the ``unit`` type: ``()``."""


UNIT_VALUE: UnitValue = UnitValue()


@dataclass(frozen=True, slots=True)
class AgentValue:
    """A first-class agent handle — opaque; not renderable or comparable."""

    name: str


# ---------------------------------------------------------------------------
# Callable value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConstructorValue:
    """A first-class constructor used as a callable value — opaque.

    Carries only the owner/variant identity needed to build a record or enum
    at the call site.  Field order and types (and concreteness) come from the
    call site's checked result type; type arguments are erased — never
    represented at runtime.  Like ``AgentValue`` it is not renderable or
    comparable by the language.

    ``nominal`` is the ``NominalId`` (module + declared name) of the owning
    type.  ``display_name`` is the user-facing name for rendering.  ``variant``
    is the enum variant name, or ``None`` for a record constructor.

    Equality and hash are by ``(nominal, variant)``; ``display_name`` is
    excluded (rendering metadata only).
    """

    nominal: NominalId
    display_name: str = field(compare=False, hash=False)
    variant: str | None


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


# ---------------------------------------------------------------------------
# Nominal value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecordValue:
    """A record-typed value.

    ``nominal`` is the ``NominalId`` (module + declared name) — the identity
    key.  ``display_name`` is the user-facing name for rendering and
    diagnostics; it is excluded from equality and hash.  ``fields`` holds
    the record's field values.

    Equality and hash are by ``(nominal, fields)``; ``display_name`` is
    excluded (rendering metadata only, mirroring how ``RecordType`` excludes
    ``fields`` from its own equality).
    """

    nominal: NominalId
    display_name: str = field(compare=False, hash=False)
    fields: dict[str, Value] = field(default_factory=dict)

    def __hash__(self) -> int:
        # Use hash(v) rather than repr(v) so that the eq/hash contract holds:
        # equal values (e.g. JsonValue(1) == JsonValue(Decimal("1.0"))) hash the same.
        return hash(
            (self.nominal, tuple(sorted((k, hash(v)) for k, v in self.fields.items())))
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RecordValue):
            return self.nominal == other.nominal and self.fields == other.fields
        return NotImplemented


@dataclass(frozen=True, slots=True)
class EnumValue:
    """An enum-typed value: the active variant name plus any payload fields.

    ``nominal`` is the ``NominalId`` (module + declared name) — the identity
    key.  ``display_name`` is the user-facing name for rendering and
    diagnostics; it is excluded from equality and hash.  ``variant`` is the
    active variant name.  ``fields`` holds the variant's payload field values.

    Equality and hash are by ``(nominal, variant, fields)``; ``display_name``
    is excluded (rendering metadata only).
    """

    nominal: NominalId
    display_name: str = field(compare=False, hash=False)
    variant: str
    fields: dict[str, Value] = field(default_factory=dict)

    def __hash__(self) -> int:
        # Use hash(v) rather than repr(v) so that the eq/hash contract holds.
        return hash(
            (
                self.nominal,
                self.variant,
                tuple(sorted((k, hash(v)) for k, v in self.fields.items())),
            )
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EnumValue):
            return (
                self.nominal == other.nominal
                and self.variant == other.variant
                and self.fields == other.fields
            )
        return NotImplemented


@dataclass(frozen=True, slots=True)
class ExceptionValue:
    """A built-in AgL exception value.

    ``nominal`` is the ``NominalId`` (module + declared name) — the identity
    key.  Built-in exceptions use ``NominalId(PRELUDE_ID, name)``.
    ``display_name`` is the user-facing exception class name (e.g.
    ``"AgentParseError"``); it is excluded from equality and hash.
    ``fields`` maps the exception's declared field names to their values.
    The ``"message"`` and ``"trace_id"`` fields are always present (base
    ``Exception`` contract).

    Equality and hash are by ``(nominal, fields)``; ``display_name`` is
    excluded (rendering metadata only).
    """

    nominal: NominalId
    display_name: str = field(compare=False, hash=False)
    fields: dict[str, Value] = field(default_factory=dict)

    def __hash__(self) -> int:
        # Use hash(v) rather than repr(v) so that the eq/hash contract holds.
        return hash(
            (self.nominal, tuple(sorted((k, hash(v)) for k, v in self.fields.items())))
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ExceptionValue):
            return self.nominal == other.nominal and self.fields == other.fields
        return NotImplemented


@dataclass(frozen=True, slots=True)
class IrClosureValue:
    """An IR closure: function_id plus its captured environment."""

    function_id: FunctionId
    captures: tuple[tuple[SymbolId, Slot], ...]
    param_labels: tuple[str, ...] = ()
    arity: int = 0
    result_label: str = "?"

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


# ---------------------------------------------------------------------------
# Broad runtime value union
# ---------------------------------------------------------------------------

Value: TypeAlias = (
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
    | ConstructorValue
    | IrClosureValue
)

# ---------------------------------------------------------------------------
# Frame and cell model (D5)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Cell:
    """A mutable box wrapping a ``Value``.

    Used as the slot for ``var`` (mutable) bindings in the per-invocation
    frame.  The cell itself is mutable (not frozen) so that ``IrAssign`` can
    update the contained value in place.

    Closures capture a ``var`` by capturing the ``Cell`` reference; the cell
    is allocated fresh each time ``IrBind`` executes for a ``var`` symbol (D5).
    """

    value: Value


#: A slot in the runtime frame is either a ``Value`` (for ``let`` bindings)
#: or a ``Cell`` (for ``var`` bindings).  Discriminate with ``isinstance``.
Slot = Value | Cell

#: Runtime frame type: maps each bound ``SymbolId`` to its slot.
Frame = dict[SymbolId, Slot]

__all__ = [
    "UNIT_VALUE",
    "AgentValue",
    "BoolValue",
    "Cell",
    "ConstructorValue",
    "DecimalValue",
    "DictValue",
    "EnumValue",
    "ExceptionValue",
    "Frame",
    "IntValue",
    "IrClosureValue",
    "JsonValue",
    "ListValue",
    "NominalId",
    "RecordValue",
    "Slot",
    "TextValue",
    "UnitValue",
    "Value",
    "_json_eq",
    "_json_hash",
]
