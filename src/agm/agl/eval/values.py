"""AgL runtime value hierarchy — full union including container and nominal types.

The leaf value tags (``TextValue``, ``IntValue``, etc.) live in the canonical,
frontend-free home ``agm.agl.values``; this module re-exports them for backward
compatibility so that all existing ``from agm.agl.eval.values import ...`` sites
keep working unchanged.

Container types (``ListValue``, ``DictValue``) and nominal types
(``RecordValue``, ``EnumValue``, ``ExceptionValue``) whose payloads reference
the broad ``Value`` union are defined HERE.  They migrate to ``agm.agl.values``
in Milestone M4 when the closure/constructor forms become AST-free.

``Closure`` and ``ConstructorValue`` are AST/Type/Scope-coupled and remain
defined here until Milestone M4.

The **broad** ``Value`` union defined here includes container/nominal types,
``Closure`` and ``ConstructorValue`` in addition to the leaf tags; the **narrow**
union in ``agm.agl.values`` has only the leaf tags.  Both are intentional during
the migration period and collapse in M4/M9.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

# ---------------------------------------------------------------------------
# Re-export all leaf tags and helpers from the canonical home
# ---------------------------------------------------------------------------
from agm.agl.values import (
    UNIT_VALUE,
    AgentValue,
    BoolValue,
    DecimalValue,
    IntValue,
    JsonValue,
    TextValue,
    UnitValue,
    _json_eq,
    _json_hash,
)
from agm.agl.values import (
    Value as BaseValue,
)

if TYPE_CHECKING:
    from agm.agl.eval.scope import Scope
    from agm.agl.syntax.nodes import Expr
    from agm.agl.typecheck.types import Type

# ---------------------------------------------------------------------------
# AST/Type/Scope-coupled value types (stay here until M4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConstructorValue:
    """A first-class constructor used as a callable value — opaque.

    Carries only the owner/variant identity needed to build a record or enum
    at the call site.  Field order and types (and concreteness) come from the
    call site's checked result type; type arguments are erased — never
    represented at runtime.  Like ``AgentValue`` it is not renderable or
    comparable by the language.

    ``owner_name`` is the owning type name; ``variant`` is the enum variant
    name, or ``None`` for a record constructor.
    """

    owner_name: str
    variant: str | None


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
# Broad Value union (forward-declared for container field annotations)
# ---------------------------------------------------------------------------

# The broad union is defined after the container/nominal classes below.
# We use a forward reference string in their field annotations so Python
# does not complain at class-body time (from __future__ import annotations
# makes all annotations strings anyway).

# ---------------------------------------------------------------------------
# Container value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ListValue:
    """A ``list[T]`` value: an immutable tuple of ``Value`` items."""

    elements: "tuple[Value, ...]"


@dataclass(frozen=True, slots=True)
class DictValue:
    """A ``dict[text, V]`` value: an immutable mapping of str → Value."""

    # Stored as a plain dict; frozen by convention (no mutation after creation).
    entries: "dict[str, Value]" = field(default_factory=dict)

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
    """A record-typed value."""

    type_name: str
    fields: "dict[str, Value]" = field(default_factory=dict)

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
    fields: "dict[str, Value]" = field(default_factory=dict)

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
    fields: "dict[str, Value]" = field(default_factory=dict)

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
# Broad Value union (legacy interpreter union — includes all types)
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
    | Closure
)

__all__ = [
    "UNIT_VALUE",
    "AgentValue",
    "BaseValue",
    "BoolValue",
    "Closure",
    "ConstructorValue",
    "DecimalValue",
    "DictValue",
    "EnumValue",
    "ExceptionValue",
    "IntValue",
    "JsonValue",
    "ListValue",
    "RecordValue",
    "TextValue",
    "UnitValue",
    "Value",
    "_json_eq",
    "_json_hash",
]
