"""Syntactic type-expression nodes for the AgL AST.

Every TypeExpr node is an immutable frozen dataclass.  ``span`` and
``node_id`` are always present but excluded from equality/hashing so that two
structurally identical type expressions compare equal regardless of where they
appear in the source.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agm.agl.syntax.spans import SourceSpan

if TYPE_CHECKING:
    from agm.agl.syntax.nodes import QualifierChain

# The builtin type spellings that lex as a plain ``NAME`` in type position
# (the parser maps them to primitive/container TypeExpr nodes — see the
# ``prim_or_name``/``applied_type`` transformers).  ``agent`` is excluded: it is
# a reserved keyword, not a NAME.  Consumers that classify identifiers without a
# parse — notably the REPL syntax highlighter — use this set to recognise the
# builtin types case-faithfully (identifier capitalization carries no meaning).
BUILTIN_TYPE_NAMES: frozenset[str] = frozenset(
    {"text", "json", "bool", "int", "decimal", "unit", "array", "dict"}
)

# The type-parameter wildcard occupies a source-level slot without introducing
# a type variable.  The AST retains every slot for later declaration-specific
# interpretation (notably method receivers), while semantic binding sites use
# ``type_parameter_bindings``.
TYPE_PARAMETER_WILDCARD = "_"


def type_parameter_bindings(type_params: tuple[str, ...]) -> tuple[str, ...]:
    """Return the type parameters that introduce readable type variables."""
    return tuple(param for param in type_params if param != TYPE_PARAMETER_WILDCARD)


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


class ImportMode(enum.Enum):
    """Determines which names are imported from the module."""

    ALL = "ALL"
    USING = "USING"
    HIDING = "HIDING"


@dataclass(frozen=True, slots=True)
class NameT:
    """A named type reference (record, enum, or type-alias name)."""

    name: str
    span: SourceSpan = field(compare=False)
    node_id: int = field(compare=False)
    qualifier: QualifierChain | None = None


@dataclass(frozen=True, slots=True)
class ArrayT:
    """An ``array[T]`` type."""

    elem: TypeExpr
    span: SourceSpan = field(compare=False)
    node_id: int = field(compare=False)


@dataclass(frozen=True, slots=True)
class DictT:
    """A ``dict[text, V]`` type.  Dict keys are always ``text`` in AgL."""

    value: TypeExpr
    span: SourceSpan = field(compare=False)
    node_id: int = field(compare=False)


@dataclass(frozen=True, slots=True)
class UnitT:
    """The ``unit`` primitive type — the type of side-effecting expressions."""

    span: SourceSpan = field(compare=False)
    node_id: int = field(compare=False)


@dataclass(frozen=True, slots=True)
class AgentT:
    """The ``agent`` opaque type.  Agent values are first-class but not JSON-shaped."""

    span: SourceSpan = field(compare=False)
    node_id: int = field(compare=False)


@dataclass(frozen=True, slots=True)
class ReceiverType:
    """The implicit receiver type of an unannotated first ``self`` parameter."""

    span: SourceSpan = field(compare=False)
    node_id: int = field(compare=False)


@dataclass(frozen=True, slots=True)
class FuncT:
    """A function type ``(A, B) -> C`` — positional parameters only in the type.

    ``params`` is the ordered tuple of parameter types; ``result`` is the return type.
    Named/optional arguments are erased from the value type (they only matter at
    declared-name call sites).
    """

    params: tuple[TypeExpr, ...]
    result: TypeExpr
    span: SourceSpan = field(compare=False)
    node_id: int = field(compare=False)


@dataclass(frozen=True, slots=True)
class AppliedT:
    """A type application ``Name[args]`` or qualified ``Name[args]``."""

    name: str
    args: tuple[TypeExpr, ...]
    span: SourceSpan = field(compare=False)
    node_id: int = field(compare=False)
    qualifier: QualifierChain | None = None


# Closed union of all type-expression nodes.
TypeExpr = (
    TextT
    | JsonT
    | BoolT
    | IntT
    | DecimalT
    | NameT
    | ArrayT
    | DictT
    | UnitT
    | AgentT
    | ReceiverType
    | FuncT
    | AppliedT
)
