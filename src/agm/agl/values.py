"""AgL leaf runtime value tags — canonical, frontend-free home (M1-A).

This module is the single source of truth for the **leaf runtime value tags**:
the closed set of primitive value kinds that can appear in a running AgL program
and whose payloads never contain another AgL ``Value``.

Design constraints
------------------
- Imports ONLY from the Python standard library (``decimal``, ``dataclasses``,
  ``typing``).  Must NOT import from ``eval``, ``syntax``, ``scope``,
  ``typecheck``, or ``runtime`` — not even under ``TYPE_CHECKING``.
- All members are frozen dataclasses with ``__slots__`` for memory efficiency.
- ``Value`` here is the **narrow** union (leaf tags only).  The **broad** union
  that includes container/nominal types, ``Closure`` and ``ConstructorValue``
  lives in ``agm.agl.eval.values`` during the migration period and collapses
  here in M4.
"""

from __future__ import annotations

import decimal
from dataclasses import dataclass
from typing import TypeAlias

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
# Narrow Value union (leaf tags only — no containers, no Closure, no ConstructorValue)
# ---------------------------------------------------------------------------

Value: TypeAlias = (
    TextValue
    | IntValue
    | DecimalValue
    | BoolValue
    | JsonValue
    | UnitValue
    | AgentValue
)
