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


@dataclass(frozen=True, slots=True)
class JsonValue:
    """A ``json`` value: any JSON-shaped Python object (the dynamic boundary).

    The ``raw`` field is ``object`` to allow ``None``, dicts, lists, strings,
    ints, floats, and bools — exactly what ``json.loads`` can return.  All
    operations on the payload use ``isinstance`` guards, never bare ``Any``
    access.
    """

    raw: object


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
        # Make hashable for use in sets/dict keys; tuple of sorted items.
        return hash(tuple(sorted(self.entries.items())))

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
        return hash((self.type_name, tuple(sorted(self.fields.items()))))

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
        return hash((self.type_name, self.variant, tuple(sorted(self.fields.items()))))

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
        return hash((self.type_name, tuple(sorted(self.fields.items()))))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ExceptionValue):
            return self.type_name == other.type_name and self.fields == other.fields
        return NotImplemented


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
)
