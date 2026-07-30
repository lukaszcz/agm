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
# a type variable.  A declaration retains every slot for later
# declaration-specific interpretation (notably method receivers), while
# semantic binding sites read its derived ``type_params``.
TYPE_PARAMETER_WILDCARD = "_"


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


def render_type_expr(type_expr: TypeExpr, *, parenthesize_function: bool = False) -> str:
    """Render a canonical, source-level spelling of a syntactic type expression."""
    if isinstance(type_expr, TextT):
        return "text"
    if isinstance(type_expr, JsonT):
        return "json"
    if isinstance(type_expr, BoolT):
        return "bool"
    if isinstance(type_expr, IntT):
        return "int"
    if isinstance(type_expr, DecimalT):
        return "decimal"
    if isinstance(type_expr, UnitT):
        return "unit"
    if isinstance(type_expr, AgentT):
        return "agent"
    if isinstance(type_expr, ReceiverType):
        return "self"
    if isinstance(type_expr, ArrayT):
        return f"array[{render_type_expr(type_expr.elem)}]"
    if isinstance(type_expr, DictT):
        return f"dict[text, {render_type_expr(type_expr.value)}]"
    if isinstance(type_expr, FuncT):
        if not type_expr.params:
            params = "()"
        elif len(type_expr.params) == 1:
            params = render_type_expr(type_expr.params[0], parenthesize_function=True)
        else:
            params = f"({', '.join(render_type_expr(param) for param in type_expr.params)})"
        rendered = f"{params} -> {render_type_expr(type_expr.result)}"
        return f"({rendered})" if parenthesize_function else rendered
    if isinstance(type_expr, (NameT, AppliedT)):
        qualifier = type_expr.qualifier
        prefix = ""
        if qualifier is not None:
            anchor = "" if qualifier.anchor is None else qualifier.anchor.value
            segments = "::".join(
                segment.name
                + (
                    "[" + ", ".join(render_type_expr(arg) for arg in segment.type_args) + "]"
                    if segment.type_args is not None
                    else ""
                )
                for segment in qualifier.segments
            )
            prefix = f"{anchor}{segments}" if not segments else f"{anchor}{segments}::"
        args = (
            "[" + ", ".join(render_type_expr(arg) for arg in type_expr.args) + "]"
            if isinstance(type_expr, AppliedT)
            else ""
        )
        return f"{prefix}{type_expr.name}{args}"
    raise AssertionError(f"unexpected type expression: {type_expr!r}")
