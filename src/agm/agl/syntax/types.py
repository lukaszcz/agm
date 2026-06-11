"""Syntactic type-expression nodes for the AgL AST.

Every TypeExpr node is an immutable frozen dataclass.  ``span`` and
``node_id`` are always present but excluded from equality/hashing so that two
structurally identical type expressions compare equal regardless of where they
appear in the source.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agm.agl.syntax.spans import SourceSpan


@dataclass(frozen=True, slots=True)
class TextT:
    """The ``text`` primitive type."""

    span: SourceSpan = field(compare=False)
    node_id: int = field(compare=False)


@dataclass(frozen=True, slots=True)
class JsonT:
    """The ``json`` primitive type (any JSON value)."""

    span: SourceSpan = field(compare=False)
    node_id: int = field(compare=False)


@dataclass(frozen=True, slots=True)
class BoolT:
    """The ``bool`` primitive type."""

    span: SourceSpan = field(compare=False)
    node_id: int = field(compare=False)


@dataclass(frozen=True, slots=True)
class IntT:
    """The ``int`` primitive type."""

    span: SourceSpan = field(compare=False)
    node_id: int = field(compare=False)


@dataclass(frozen=True, slots=True)
class DecimalT:
    """The ``decimal`` primitive type (exact fixed-point)."""

    span: SourceSpan = field(compare=False)
    node_id: int = field(compare=False)


@dataclass(frozen=True, slots=True)
class NameT:
    """A named type reference (record, enum, or type-alias name)."""

    name: str
    span: SourceSpan = field(compare=False)
    node_id: int = field(compare=False)


@dataclass(frozen=True, slots=True)
class ListT:
    """A ``list[T]`` type."""

    elem: TypeExpr
    span: SourceSpan = field(compare=False)
    node_id: int = field(compare=False)


@dataclass(frozen=True, slots=True)
class DictT:
    """A ``dict[text, V]`` type.  Dict keys are always ``text`` in AgL v1."""

    value: TypeExpr
    span: SourceSpan = field(compare=False)
    node_id: int = field(compare=False)


# Closed union of all type-expression nodes.
TypeExpr = TextT | JsonT | BoolT | IntT | DecimalT | NameT | ListT | DictT
