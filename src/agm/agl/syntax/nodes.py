"""Full AgL AST node set.

Every node is:
  - ``@dataclass(frozen=True, slots=True)`` — immutable and memory-efficient.
  - Carries ``span: SourceSpan`` and ``node_id: int`` as ``dc_field(compare=False)``
    so that equality/hashing are purely structural (two nodes with the same
    shape but different source locations compare equal).
  - Child collections are ``tuple`` (never ``list``).

``node_id`` is assigned by the AST builder (parser pass), not here.

Union aliases
-------------
``Stmt``, ``Expr``, ``Pattern``, ``TemplateSegment`` are closed typed unions
over their respective node families.  They are defined at the bottom of this
module after all constituent classes.
"""

from __future__ import annotations

import decimal
import enum
from dataclasses import dataclass
from dataclasses import field as dc_field

from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import TypeExpr

# ---------------------------------------------------------------------------
# Sentinel for the else-branch of IfStmt
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
# Call options (agent invocation modifiers)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AbortPolicy:
    """Parse policy: abort immediately on a parse failure."""

    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Parse policy: retry up to ``extra`` additional times on parse failure."""

    extra: int
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


ParsePolicy = AbortPolicy | RetryPolicy


@dataclass(frozen=True, slots=True)
class CallOptions:
    """Options that modify an agent-call invocation.

    ``format``       — output format hint (e.g. ``"json"``, ``"text"``).
    ``strict_json``  — override the runtime default for strict JSON parsing.
    ``parse_policy`` — what to do when the output cannot be parsed.
    """

    format: str | None
    strict_json: bool | None
    parse_policy: ParsePolicy | None
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


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

    ``render`` holds the BARE renderer name with no leading colon
    (e.g. ``"raw"``, ``"json"``, ``"bullets"``); ``None`` means default
    rendering.
    """

    expr: Expr
    render: str | None
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
class AgentCall:
    """An agent invocation: ``ask[options] template``."""

    agent: str
    options: CallOptions
    template: Template
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class NamedArg:
    """A named argument in a constructor call: ``name: value``."""

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
class CaseExprBranch:
    """A single branch in a ``case`` expression."""

    pattern: Pattern
    body: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class CaseExpr:
    """A ``case expr of { ... }`` expression."""

    subject: Expr
    branches: tuple[CaseExprBranch, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


# --- Literals ---


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
Expr = (
    VarRef
    | FieldAccess
    | Template
    | AgentCall
    | Constructor
    | BinaryOp
    | UnaryNot
    | UnaryNeg
    | IsTest
    | CaseExpr
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
# Statement nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LetDecl:
    """``let name [: type] = expr`` — immutable binding."""

    name: str
    type_ann: TypeExpr | None
    value: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class VarDecl:
    """``var name [: type] = expr`` — mutable binding."""

    name: str
    type_ann: TypeExpr | None
    value: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class SetStmt:
    """``set target = expr`` — assignment to a mutable variable."""

    target: str
    value: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class PassStmt:
    """``pass`` — no-op statement."""

    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class PrintStmt:
    """``print expr`` — emit a value to the trace output."""

    value: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class ExprStmt:
    """An expression used as a statement (result discarded)."""

    expr: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class DoUntil:
    """``do[limit] { body } until condition`` — bounded loop."""

    limit: int | None
    body: tuple[Stmt, ...]
    condition: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class IfBranch:
    """A single branch of an ``if`` statement.

    ``cond`` is either an ``Expr`` (for the ``if``/``elif`` arms) or the
    singleton ``ELSE`` sentinel (for the ``else`` arm).
    """

    cond: Expr | ElseSentinel
    body: tuple[Stmt, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class IfStmt:
    """``if … elif … else …`` statement, represented as a list of branches."""

    branches: tuple[IfBranch, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class CaseStmtBranch:
    """A single branch in a ``case`` statement."""

    pattern: Pattern
    body: tuple[Stmt, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class CaseStmt:
    """``case expr of { ... }`` statement."""

    subject: Expr
    branches: tuple[CaseStmtBranch, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class CatchClause:
    """A ``catch`` handler in a ``try/catch`` statement.

    ``exc_type`` is the exception type name (or ``None`` for a catch-all).
    ``binding`` is the optional variable name for the exception value.
    """

    exc_type: str | None
    binding: str | None
    body: tuple[Stmt, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class TryCatch:
    """``try { body } catch { handlers }`` statement."""

    body: tuple[Stmt, ...]
    handlers: tuple[CatchClause, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class Raise:
    """``raise expr`` — throw an AgL exception."""

    exc: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


# ---------------------------------------------------------------------------
# Declaration nodes (top-level constructs)
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
class ParamDecl:
    """``param name[: TypeExpr] [= expr]`` declaration.

    The type ``annotation`` is optional: ``param spec`` is equivalent to
    ``param spec: text``.  The default (``text``) is applied by the TYPECHECK
    pass, not synthesized by the parser, so ``annotation`` is ``None`` when the
    source omits it.

    The ``default`` expression is optional; ``None`` when omitted.  Evaluation
    of the default is handled by Milestone 2+ passes, not here.
    """

    name: str
    annotation: TypeExpr | None
    default: Expr | None
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class ProgramDecl:
    """``program NAME`` declaration.

    Declares the program name used for config lookup and CLI integration.
    """

    name: str
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class AgentDecl:
    """``agent NAME [= "runner string"]`` declaration.

    ``runner`` is the optional static runner-command hint (a literal string
    with NO interpolation); ``None`` for a bare declaration.
    """

    name: str
    runner: str | None
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


# ---------------------------------------------------------------------------
# Closed statement union (defined after all Stmt classes)
# ---------------------------------------------------------------------------

Stmt = (
    LetDecl
    | VarDecl
    | SetStmt
    | PassStmt
    | PrintStmt
    | ExprStmt
    | DoUntil
    | IfStmt
    | CaseStmt
    | TryCatch
    | Raise
    | RecordDef
    | EnumDef
    | TypeAlias
    | ParamDecl
    | ProgramDecl
    | AgentDecl
)


# ---------------------------------------------------------------------------
# Program root
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Program:
    """Root node of an AgL program.  ``body`` is the ordered top-level sequence."""

    body: tuple[Stmt, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
