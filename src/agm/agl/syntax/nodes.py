"""Full AgL v2 AST node set.

Every node is:
  - ``@dataclass(frozen=True, slots=True)`` — immutable and memory-efficient.
  - Carries ``span: SourceSpan`` and ``node_id: int`` as ``dc_field(compare=False)``
    so that equality/hashing are purely structural (two nodes with the same
    shape but different source locations compare equal).
  - Child collections are ``tuple`` (never ``list``).

``node_id`` is assigned by the AST builder (parser pass), not here.

Union aliases
-------------
``Expr``, ``Item``, ``Binder``, ``Declaration``, ``Pattern``, ``TemplateSegment``
are closed typed unions over their respective node families.  They are defined at
the bottom of this module after all constituent classes.

v2 design notes
---------------
- The statement category is removed: every former statement is an expression.
- ``Block`` is the sequencing expression; its value is the last item.
- ``If``, ``Case``, ``Do``, ``Try`` unify the former statement/expression variants.
- ``Call`` is the single call node for both paren-form and single-arg sugar.
- ``FuncDef`` / ``Lambda`` / ``Param`` support first-class recursive functions.
- ``UnitLit`` is the ``()`` unit-value literal.
- ``Raise`` is an expression with bottom type (usable anywhere an ``Expr`` is).
"""

from __future__ import annotations

import decimal
import enum
from dataclasses import dataclass
from dataclasses import field as dc_field
from decimal import Decimal

from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import TypeExpr

# ---------------------------------------------------------------------------
# Sentinel for the else-branch of If
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ElseSentinel:
    """Singleton sentinel used as ``IfBranch.cond`` to mark an else branch.

    Use the module-level singleton ``ELSE`` rather than constructing new
    instances.
    """


ELSE: ElseSentinel = ElseSentinel()


# ---------------------------------------------------------------------------
# Binary operator enum
# ---------------------------------------------------------------------------


class BinOp(enum.Enum):
    """Closed set of binary operators recognised by AgL."""

    EQ = "="
    NEQ = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    IN = "in"
    AND = "and"
    OR = "or"
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"


# ---------------------------------------------------------------------------
# Template segments
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextSegment:
    """A literal text fragment inside a template string."""

    text: str
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class InterpSegment:
    """An interpolated expression inside a template string (``${expr}``).

    ``expr`` is an arbitrary expression; rendering is always uniform — text
    verbatim, scalars as plain text, structured values as pretty JSON.  There
    is no ``as <renderer>`` override: the grammar accepts only ``${expr}``.
    """

    expr: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


TemplateSegment = TextSegment | InterpSegment


# ---------------------------------------------------------------------------
# Expression nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VarRef:
    """Reference to a variable or input binding."""

    name: str
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class FieldAccess:
    """``obj.field`` — member access on a record value."""

    obj: Expr
    field: str
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class Template:
    """A template string: a sequence of text and interpolation segments."""

    segments: tuple[TemplateSegment, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class NamedArg:
    """A named argument in a constructor or call expression: ``name: value``."""

    name: str
    value: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class Constructor:
    """A record or enum-variant constructor call.

    ``qualifier`` is the enum type name when the variant is namespaced
    (``Color.Red``); ``None`` when unqualified.
    """

    qualifier: str | None
    name: str
    args: tuple[NamedArg, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class BinaryOp:
    """A binary operation: ``left op right``."""

    op: BinOp
    left: Expr
    right: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class UnaryNot:
    """Logical negation: ``not operand``."""

    operand: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class UnaryNeg:
    """Arithmetic negation: ``-operand``."""

    operand: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class IsTest:
    """Pattern membership test: ``expr is [not] [Qualifier.]Variant``."""

    expr: Expr
    qualifier: str | None
    variant: str
    negated: bool
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class Call:
    """A uniform function/built-in call: ``callee(args, name: v)``.

    Also produced by the single-arg juxtaposition sugar ``f x``
    (which desugars to ``Call(callee=f, args=(x,), named_args=())``.
    """

    callee: Expr
    args: tuple[Expr, ...]
    named_args: tuple[NamedArg, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class Param:
    """A function or lambda parameter: ``name: TypeExpr [= default]``."""

    name: str
    type_expr: TypeExpr
    default: Expr | None
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class FuncDef:
    """``def name(params) -> RetType = body`` — a top-level function declaration.

    ``return_type`` is always required for ``def`` (full annotation in v1).
    ``body`` is an expression (which may be a ``Block`` for multi-step bodies).
    """

    name: str
    params: tuple[Param, ...]
    return_type: TypeExpr
    body: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class Lambda:
    """``fn(params) (-> R)? => body`` — an anonymous function expression.

    ``return_type`` is ``None`` when omitted (inferred from the body).
    Lambda parameter types are always required in v1.
    """

    params: tuple[Param, ...]
    return_type: TypeExpr | None
    body: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class Block:
    """An expression block: a sequence of items whose value is the last item.

    Items may be declarations (``FuncDef``, ``RecordDef``, …), binders
    (``LetDecl``, ``VarDecl``, ``SetStmt``), or expressions.  A block ending
    in a binder is a static error (the binder needs a continuation expression).
    """

    items: tuple[Item, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class IfBranch:
    """A single branch in an ``if`` expression.

    ``cond`` is either an ``Expr`` (condition arm) or the singleton ``ELSE``
    sentinel (the else arm).  ``body`` is a single expression (including
    ``Block`` for multi-statement bodies).
    """

    cond: Expr | ElseSentinel
    body: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class If:
    """``if cond => body | ... | else => body`` expression.

    Unifies the former ``IfStmt`` and ``IfExpr``.  An ``if`` with no ``else``
    branch yields ``unit``.
    """

    branches: tuple[IfBranch, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class CaseBranch:
    """A single branch in a ``case`` expression."""

    pattern: Pattern
    body: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class Case:
    """``case expr of { ... }`` expression.

    Unifies the former ``CaseStmt`` and ``CaseExpr``.
    """

    subject: Expr
    branches: tuple[CaseBranch, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class Do:
    """``do[limit] body until condition`` — bounded loop expression.

    Yields ``unit``.  ``limit`` is the optional iteration bound (``None``
    means no static bound is enforced).  ``body`` is typically a ``Block``.
    """

    limit: int | None
    body: Expr
    condition: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class CatchClause:
    """A ``catch`` handler in a ``try`` expression.

    ``exc_type`` is the exception type name (or ``None`` for a catch-all).
    ``binding`` is the optional variable name for the exception value.
    ``body`` is a single expression (may be a ``Block``).
    """

    exc_type: str | None
    binding: str | None
    body: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class Try:
    """``try body catch { handlers }`` expression.

    Unifies the former ``TryCatch``.  The type is the unified type of the
    body and all handler bodies.
    """

    body: Expr
    handlers: tuple[CatchClause, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class Raise:
    """``raise expr`` — throw an AgL exception.

    Has the bottom type: it is assignable to any expected type because it
    never produces a value.
    """

    exc: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


# --- Literals ---


@dataclass(frozen=True, slots=True)
class UnitLit:
    """The ``()`` unit literal — the single value of the ``unit`` type."""

    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class IntLit:
    """An integer literal."""

    value: int
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class DecimalLit:
    """A decimal (fixed-point) literal.  Always stored as ``decimal.Decimal``."""

    value: decimal.Decimal
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class BoolLit:
    """A boolean literal (``true`` or ``false``)."""

    value: bool
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class NullLit:
    """The ``null`` literal."""

    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class StringLit:
    """A plain (non-interpolated) string literal."""

    value: str
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class ListLit:
    """A list literal: ``[e1, e2, ...]``."""

    elements: tuple[Expr, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class DictEntry:
    """A single key/value entry in a dict literal."""

    key: StringLit
    value: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class DictLit:
    """A dict literal: ``{k: v, ...}``."""

    entries: tuple[DictEntry, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


# Closed union of all expression nodes.
# NOTE: Raise is an Expr (bottom type — assignable to any expected type).
# Block, If, Case, Do, Try are expressions (value-producing in v2).
# Let/Var/Set are NOT Expr — they are binders (Item only, not directly usable
# in expression position; they scope over the rest of a Block).
Expr = (
    VarRef
    | FieldAccess
    | Template
    | Constructor
    | BinaryOp
    | UnaryNot
    | UnaryNeg
    | IsTest
    | Call
    | Lambda
    | Block
    | If
    | Case
    | Do
    | Try
    | Raise
    | UnitLit
    | IntLit
    | DecimalLit
    | BoolLit
    | NullLit
    | StringLit
    | ListLit
    | DictLit
)


# ---------------------------------------------------------------------------
# Pattern nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WildcardPattern:
    """The ``_`` wildcard pattern (matches anything)."""

    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class LiteralPattern:
    """A literal-value pattern (matches a specific literal)."""

    literal: IntLit | DecimalLit | BoolLit | StringLit | NullLit
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class VarPattern:
    """A binding pattern — captures the matched value into ``name``."""

    name: str
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class PatternField:
    """A named field sub-pattern in a constructor pattern."""

    name: str
    pattern: Pattern
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class ConstructorPattern:
    """A constructor (record/variant) destructuring pattern."""

    qualifier: str | None
    name: str
    fields: tuple[PatternField, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


# Closed union of all pattern nodes.
Pattern = WildcardPattern | LiteralPattern | VarPattern | ConstructorPattern


# ---------------------------------------------------------------------------
# Binder nodes (block-item level, not independently usable as Expr)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LetDecl:
    """``let name [: type] = expr`` — immutable binding (scopes over continuation)."""

    name: str
    type_ann: TypeExpr | None
    value: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class VarDecl:
    """``var name [: type] = expr`` — mutable binding (scopes over continuation)."""

    name: str
    type_ann: TypeExpr | None
    value: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class SetStmt:
    """``set target = expr`` — assignment to a mutable variable.  Yields ``unit``."""

    target: str
    value: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


# Closed union of binder nodes.
Binder = LetDecl | VarDecl | SetStmt


# ---------------------------------------------------------------------------
# Declaration nodes (top-level + block-level constructs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FieldDef:
    """A field definition in a ``record`` or enum-variant body."""

    name: str
    type_expr: TypeExpr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class RecordDef:
    """``record Name { fields }`` declaration."""

    name: str
    fields: tuple[FieldDef, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class VariantDef:
    """A single variant inside an ``enum`` declaration."""

    name: str
    fields: tuple[FieldDef, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class EnumDef:
    """``enum Name { variants }`` declaration."""

    name: str
    variants: tuple[VariantDef, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class TypeAlias:
    """``type Name = TypeExpr`` declaration."""

    name: str
    type_expr: TypeExpr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class InputDecl:
    """``input name[: TypeExpr]`` declaration.

    The type ``annotation`` is optional: ``input spec`` is equivalent to
    ``input spec: text``.  The default (``text``) is applied by the TYPECHECK
    pass, not synthesized by the parser, so ``annotation`` is ``None`` when the
    source omits it.
    """

    name: str
    annotation: TypeExpr | None
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class AgentDecl:
    """``agent NAME [= "runner string"]`` declaration.

    ``runner`` is the optional static runner-command hint (a literal string
    with NO interpolation); ``None`` for a bare declaration.
    In v2, agent names are ordinary value bindings of type ``agent``.
    """

    name: str
    runner: str | None
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


# ---------------------------------------------------------------------------
# Config pragma
# ---------------------------------------------------------------------------

#: The set of value types a config pragma may carry.
PragmaValue = bool | int | Decimal | str


@dataclass(frozen=True, slots=True)
class ConfigPragma:
    """``config KEY = VALUE`` header pragma.

    Must appear before any non-pragma item at the program root.
    Enforced by the scope pass; grammatically it is a top-level item.

    ``key``    — the pragma name (e.g. ``"log"``, ``"max_iters"``).
    ``value``  — a statically-known scalar: ``bool``, ``int``,
                 ``Decimal``, or ``str``.
    """

    key: str
    value: PragmaValue
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


# Closed union of declaration nodes.
# FuncDef is a declaration (top-level or block-level named function).
Declaration = FuncDef | RecordDef | EnumDef | TypeAlias | InputDecl | AgentDecl | ConfigPragma


# ---------------------------------------------------------------------------
# Item union — element type of Block.items
# ---------------------------------------------------------------------------

# An item is anything that can appear in a block sequence:
# declarations (introduce names), binders (scope over the rest), or expressions.
Item = Declaration | Binder | Expr


# ---------------------------------------------------------------------------
# Program root
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Program:
    """Root node of an AgL program.  ``body`` is a ``Block`` of top-level items."""

    body: Block
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
