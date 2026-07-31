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
``Expr``, ``Item``, ``Binder``, ``Declaration``, ``Pattern``, ``TemplateSegment``
are closed typed unions over their respective node families.  They are defined at
the bottom of this module after all constituent classes.

Design notes
---------------
- The statement category is removed: every former statement is an expression.
- ``Block`` is the sequencing expression; its value is the last item.
- ``If``, ``Case``, ``Loop``, ``Try`` unify the former statement/expression variants.
- ``Call`` is the single call node for both paren-form and single-arg sugar.
- ``FuncDef`` / ``Lambda`` / ``Param`` support first-class recursive functions.
- ``UnitLit`` is the ``()`` unit-value literal.
- ``Raise`` and ``Return`` are expressions with bottom type (usable anywhere an ``Expr`` is).
"""

from __future__ import annotations

import decimal
import enum
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TypeGuard

from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import TYPE_PARAMETER_WILDCARD, ImportMode, TypeExpr

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

    EQ = "=="
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


class InfixAssoc(enum.Enum):
    """Associativity declared for a user-defined infix operator."""

    LEFT = "left"
    RIGHT = "right"


# ---------------------------------------------------------------------------
# Module system nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ImportItem:
    """A selected import member: ``scope_path::name [as rename]``."""

    name: str
    rename: str | None
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
    scope_path: tuple[ScopeSegment, ...] = ()


@dataclass(frozen=True, slots=True)
class ImportDecl:
    """``[open] import MODPATH[/*] [as ALIAS] [using…|hiding…]`` declaration.

    ``scope_path`` is non-empty when the declaration is a region item: the
    region's own path, not to be confused with an ``ImportItem``'s
    ``scope_path``, which selects a member inside the *imported* module.
    """

    module_path: tuple[str, ...]
    wildcard: bool
    is_open: bool
    alias: str | None
    mode: ImportMode
    items: tuple[ImportItem, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
    scope_path: tuple[ScopeSegment, ...] = ()


@dataclass(frozen=True, slots=True)
class ExportItem:
    """A selected export member: ``scope_path::name [as rename]``."""

    name: str
    rename: str | None
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
    scope_path: tuple[ScopeSegment, ...] = ()


@dataclass(frozen=True, slots=True)
class ExportDecl:
    """``export MODPATH[/*] [using…|hiding…]`` declaration.

    ``scope_path`` is non-empty when the declaration is a region item: the
    forwarded atoms are re-rooted under it. Not to be confused with an
    ``ExportItem``'s ``scope_path``, which selects a member inside the
    *forwarded* module.
    """

    module_path: tuple[str, ...]
    wildcard: bool
    mode: ImportMode
    items: tuple[ExportItem, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
    scope_path: tuple[ScopeSegment, ...] = ()


@dataclass(frozen=True, slots=True)
class ScopeSegment:
    """One named segment of a scope path."""

    name: str
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class ScopeRef:
    """A scope reference, with an optional unambiguous slash module route."""

    module_route: tuple[str, ...]
    scope_path: tuple[ScopeSegment, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class OpenDecl:
    """``open scope_ref [using…|hiding…]`` declaration."""

    scope_ref: ScopeRef
    mode: ImportMode
    items: tuple[ImportItem, ...]
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
    """An interpolated expression inside a template string (``%{expr}``).

    ``expr`` is an arbitrary expression; interpolation renders it with the
    default program-output options (single-line, unquoted top-level text).
    There is no ``as <renderer>`` override: the grammar accepts only ``%{expr}``.
    """

    expr: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


TemplateSegment = TextSegment | InterpSegment


# ---------------------------------------------------------------------------
# Expression nodes
# ---------------------------------------------------------------------------


class QualifierAnchor(enum.Enum):
    """The explicit anchor of an expression qualifier chain."""

    MODULE = "/"
    CURRENT_MODULE = "::"


@dataclass(frozen=True, slots=True)
class QualifierSegment:
    """One qualifier-chain segment, optionally applied to type arguments.

    ``anchored`` retains a slash prefix so validation can reject an anchor that
    appears after another chain prefix.
    """

    name: str
    type_args: tuple[TypeExpr, ...] | None
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
    anchored: bool = False


@dataclass(frozen=True, slots=True)
class QualifierChain:
    """Qualified reference prefix and its selected member."""

    anchor: QualifierAnchor | None
    segments: tuple[QualifierSegment, ...]
    member: str
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)

    @property
    def route_segments(self) -> tuple[str, ...]:
        """Return the slash-expanded route represented by the chain segments."""
        return tuple(part for segment in self.segments for part in segment.name.split("/"))

    @property
    def anchored(self) -> bool:
        """Whether the chain uses an absolute module anchor."""
        return self.anchor is QualifierAnchor.MODULE

    def render(self) -> str:
        """Render the chain's qualifier prefix without its selected member."""
        return ("/" if self.anchored else "") + "/".join(self.route_segments)


@dataclass(frozen=True, slots=True)
class VarRef:
    """Reference to a variable, param binding, or qualified constructor."""

    name: str
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
    qualifier: QualifierChain | None = None


@dataclass(frozen=True, slots=True)
class FieldAccess:
    """``obj.field`` — member access on a record value."""

    obj: Expr
    field: str
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class IndexAccess:
    """``obj[index]`` — index access on an array or dict value."""

    obj: Expr
    index: Expr
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
    """A named argument in a constructor or call expression: ``name = value``."""

    name: str
    value: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class RecordUpdate:
    """A functional update: ``target with field = value, ...``.

    Produces a copy of the record or exception value with the listed fields
    replaced.  Updates reuse :class:`NamedArg` for the ``field = value`` pairs.
    """

    target: Expr
    updates: tuple[NamedArg, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class Placeholder:
    """A call-argument placeholder.

    ``index`` is ``None`` for a bare placeholder and 1-based for a numbered one.
    """

    index: int | None
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
class Cast:
    """A type cast (``expr as T``) or convertibility test (``expr as? T``).

    ``test_only=False`` — the ``as`` operator; yields a value of type T.
    ``test_only=True``  — the ``as?`` operator; yields ``bool``.
    """

    expr: Expr
    target_type: TypeExpr
    test_only: bool
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class IsTest:
    """Pattern membership test: ``expr is [not] [qualifier chain]``."""

    expr: Expr
    variant: str
    negated: bool
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
    qualifier: QualifierChain | None = None


@dataclass(frozen=True, slots=True)
class TypeApply:
    """Explicit value-position type application: ``expr::[T]``."""

    expr: Expr
    type_args: tuple[TypeExpr, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class Call:
    """A uniform function/built-in call: ``callee(args, name: v)``.

    Also produced by the single-arg juxtaposition sugar ``f x``
    (which desugars to ``Call(callee=f, args=(x,), named_args=())``.

    ``type_args`` is set by the typed-call syntax ``callee::[T](args)``
    (e.g. ``ask-request::[Review](...)``); it is ``()`` for ordinary calls.
    The type arguments are static ``TypeExpr`` values resolved by the type checker —
    they are never evaluated at runtime.
    """

    callee: Expr
    args: tuple[Expr, ...]
    named_args: tuple[NamedArg, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
    type_args: tuple[TypeExpr, ...] = ()


class ParamKind(enum.Enum):
    """The zone a parameter belongs to in its parameter list.

    Values are stable strings for debuggability; no code should branch on them.
    The transformer assigns a concrete kind to every ``Param`` at parse time;
    no downstream pass ever sees a marker token.
    """

    POSITIONAL_ONLY = "positional_only"
    STANDARD = "standard"
    NAMED_ONLY = "named_only"


@dataclass(frozen=True, slots=True)
class Param:
    """A function/lambda parameter or a record/enum-variant/exception field.

    ``kind`` records which zone this parameter belongs to (positional-only,
    standard, or named-only). The transformer assigns a concrete kind to every
    parameter, and shared argument binding enforces it for calls and patterns.
    ``default`` is ``None`` for field params (records/variants/exceptions);
    only ``def``/``builtin def``/lambda params may carry a default expression.
    ``type_expr`` is ``None`` when the source omitted the annotation, which the
    builder permits only for a method's first ``self`` parameter.
    """

    name: str
    type_expr: TypeExpr | None
    kind: ParamKind
    default: Expr | None
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


# ---------------------------------------------------------------------------
# Type-parameter slots — shared by every declaration that can be generic
# ---------------------------------------------------------------------------


class GenericDeclaration:
    """A declaration carrying a source-level type-parameter slot list.

    ``type_param_slots`` keeps every slot the source wrote, including the
    ``_`` wildcard, which holds a position without introducing a type variable;
    consumers that care about positions (notably a method receiver) read it.
    Semantic binding sites read the derived ``type_params`` instead.
    """

    __slots__ = ()

    type_param_slots: tuple[str, ...]

    @property
    def type_params(self) -> tuple[str, ...]:
        """The slots that introduce readable type variables."""
        return tuple(slot for slot in self.type_param_slots if slot != TYPE_PARAMETER_WILDCARD)


@dataclass(frozen=True, slots=True)
class FuncDef(GenericDeclaration):
    """``def name(params) (-> RetType)? = body`` — a top-level function declaration.

    ``return_type`` is ``None`` when omitted and inferred from the body.
    ``body`` is an expression (which may be a ``Block`` for multi-step bodies).
    """

    name: str
    params: tuple[Param, ...]
    return_type: TypeExpr | None
    body: Expr | None
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
    type_param_slots: tuple[str, ...] = ()
    is_builtin: bool = False
    is_extern: bool = False
    scope_path: tuple[ScopeSegment, ...] = ()


@dataclass(frozen=True, slots=True)
class Lambda:
    """``fn(params) (-> R)? => body`` — an anonymous function expression.

    ``return_type`` is ``None`` when omitted (inferred from the body).
    Lambda parameter types are always required in AgL.
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
    (``LetDecl``, ``VarDecl``, ``AssignStmt``), or expressions. The block's
    value is its final item. A final ``let`` or ``var`` is allowed and yields
    ``unit`` (or bottom when its initializer exits); earlier bare expressions
    must be ``unit``- or bottom-valued.
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
    """``if cond => body | ... [|] else => body`` expression.

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


@dataclass(frozen=True, slots=True, init=False)
class Loop:
    """Unified loop expression.

    Syntax: ``(for VAR in ITER [range_tail])? (while COND)? do[BOUND]? body (until COND | done)?``.

    Yields ``unit``.

    Fields
    ------
    for_var:
        The loop variable name for a ``for VAR in ITER`` clause, or ``None``.
    for_iter:
        For a collection ``for``: the collection expression.
        For a range ``for``: the lower bound expression (``a`` in ``for i in a to b``).
        ``None`` when there is no ``for`` clause.
    for_range_to:
        The upper/lower bound expression (``b``) for an integer-range ``for``
        clause (``to b`` or ``downto b``).  ``None`` means this is a collection
        ``for`` (or there is no ``for`` clause at all).
    for_range_down:
        ``True`` for a ``downto`` range, ``False`` for a ``to`` range or when
        there is no range clause.
    for_range_by:
        The step expression for an integer-range ``for`` clause (``by k``), or
        ``None`` when the step is the default (1).  Always ``None`` when
        ``for_range_to`` is ``None``.
    while_cond:
        The ``while`` guard expression, or ``None``.
    bound:
        The optional iteration bound ``[expr]`` (an ``int``-typed expression
        evaluated once at loop entry), or ``None`` (unbounded).
    body:
        The loop body (typically a ``Block``).
    until_cond:
        The ``until`` exit condition expression, or ``None``.  ``None`` means
        ``done`` or an omitted terminator, both equivalent to ``until false``.
    """

    for_var: str | None
    for_iter: Expr | None
    for_range_to: Expr | None
    for_range_down: bool
    for_range_by: Expr | None
    while_cond: Expr | None
    bound: Expr | None
    body: Expr
    until_cond: Expr | None
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)

    def __init__(
        self,
        *,
        for_var: str | None = None,
        for_iter: Expr | None = None,
        for_range_to: Expr | None = None,
        for_range_down: bool = False,
        for_range_by: Expr | None = None,
        while_cond: Expr | None = None,
        bound: Expr | None = None,
        body: Expr,
        until_cond: Expr | None = None,
        span: SourceSpan,
        node_id: int,
        limit: int | Expr | None = None,
        condition: Expr | None = None,
    ) -> None:
        if bound is None and limit is not None:
            bound = (
                IntLit(value=limit, span=span, node_id=node_id) if isinstance(limit, int) else limit
            )
        if until_cond is None and condition is not None:
            until_cond = condition
        object.__setattr__(self, "for_var", for_var)
        object.__setattr__(self, "for_iter", for_iter)
        object.__setattr__(self, "for_range_to", for_range_to)
        object.__setattr__(self, "for_range_down", for_range_down)
        object.__setattr__(self, "for_range_by", for_range_by)
        object.__setattr__(self, "while_cond", while_cond)
        object.__setattr__(self, "bound", bound)
        object.__setattr__(self, "body", body)
        object.__setattr__(self, "until_cond", until_cond)
        object.__setattr__(self, "span", span)
        object.__setattr__(self, "node_id", node_id)

    @property
    def limit(self) -> int | Expr | None:
        if isinstance(self.bound, IntLit):
            return self.bound.value
        return self.bound

    @property
    def condition(self) -> Expr | None:
        return self.until_cond


@dataclass(frozen=True, slots=True)
class Break:
    """``break`` — exit the innermost enclosing loop immediately.

    Has the bottom type: assignable to any expected type because it never
    produces a value (control does not continue past this expression).
    """

    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class Continue:
    """``continue`` — proceed to the next iteration of the innermost enclosing loop.

    Has the bottom type: assignable to any expected type because it never
    produces a value (control does not continue past this expression).
    """

    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


Do = Loop


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


@dataclass(frozen=True, slots=True)
class Return:
    """``return [expr]`` — return early from the nearest enclosing function.

    Has the bottom type: it is assignable to any expected type because it
    never produces a value at the expression site.
    """

    value: Expr | None
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
class ArrayLit:
    """An array literal: ``[e1, e2, ...]``."""

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
# NOTE: Raise/Return are Exprs (bottom type — assignable to any expected type).
# Block, If, Case, Loop, Try are expressions (value-producing in AgL).
# Let/Var/Set are NOT Expr — they are binders (Item only, not directly usable
# in expression position; they scope over the rest of a Block).
Expr = (
    VarRef
    | FieldAccess
    | IndexAccess
    | Template
    | Placeholder
    | BinaryOp
    | UnaryNot
    | UnaryNeg
    | Cast
    | IsTest
    | TypeApply
    | Call
    | RecordUpdate
    | Lambda
    | Block
    | If
    | Case
    | Loop
    | Try
    | Raise
    | Return
    | Break
    | Continue
    | UnitLit
    | IntLit
    | DecimalLit
    | BoolLit
    | NullLit
    | StringLit
    | ArrayLit
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
    """A bare pattern name: top-level constructors or field-directed nested syntax.

    At case-pattern top level, the name denotes a nullary constructor. Within
    a constructor field, it denotes that field only when its spelling matches
    the field name; otherwise it may denote a nullary constructor of the field
    type. Use :class:`AsPattern` for an explicit general-purpose binder.
    """

    name: str
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class AsPattern:
    """A pattern annotated with a binder for its complete matched value."""

    pattern: Pattern
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
    """A constructor (record/variant) destructuring pattern.

    ``positional`` holds positional sub-patterns (in source order). They fill
    positional-capable (POSITIONAL_ONLY or STANDARD) fields left to right, or
    use named-only shorthand when no positional-capable slots remain.
    ``named`` holds named sub-patterns ``name = pattern`` (PatternField), each
    bound to the field with that name.  Positional must precede named (enforced
    by the transformer); the checker routes both through ``bind_arguments``.
    Partial patterns are allowed — unmentioned fields are wildcards.
    ``has_argument_list`` records whether the source wrote an argument list
    ``(...)`` after the name or qualifier chain — ``A::x`` and ``A::x()`` are
    distinct nodes so a bare qualified pattern can be told apart from a
    nullary qualified constructor pattern.
    """

    name: str
    positional: tuple[Pattern, ...]
    named: tuple[PatternField, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
    qualifier: QualifierChain | None = None
    has_argument_list: bool = False


# Closed union of all pattern nodes.
Pattern = WildcardPattern | LiteralPattern | VarPattern | AsPattern | ConstructorPattern


@dataclass(frozen=True, slots=True)
class PatternBinderCandidate:
    """One named pattern occurrence that may introduce a binding.

    ``nested`` distinguishes occurrences directed by a constructor field from
    roots.  Match sites supply the policy that decides whether a bare root
    binds; ``as`` occurrences always do.
    """

    name: str
    node_id: int
    span: SourceSpan
    nested: bool
    is_as_pattern: bool


def pattern_binder_candidates(pattern: Pattern) -> tuple[PatternBinderCandidate, ...]:
    """Return named binder candidates in source preorder.

    This is the shared pattern-binding seam for scope and later checked-pattern
    consumers. Wildcards and literals introduce no candidate, while every bare
    name and ``as`` name retains its nesting and source identity.
    """

    candidates: list[PatternBinderCandidate] = []

    def collect(current: Pattern, *, nested: bool) -> None:
        if isinstance(current, VarPattern):
            candidates.append(
                PatternBinderCandidate(
                    name=current.name,
                    node_id=current.node_id,
                    span=current.span,
                    nested=nested,
                    is_as_pattern=False,
                )
            )
        elif isinstance(current, AsPattern):
            collect(current.pattern, nested=nested)
            candidates.append(
                PatternBinderCandidate(
                    name=current.name,
                    node_id=current.node_id,
                    span=current.span,
                    nested=nested,
                    is_as_pattern=True,
                )
            )
        elif isinstance(current, ConstructorPattern):
            for child in current.positional:
                collect(child, nested=True)
            for field in current.named:
                collect(field.pattern, nested=True)

    collect(pattern, nested=False)
    return tuple(candidates)


def pattern_binding_node_ids(pattern: Pattern) -> tuple[int, ...]:
    """Return source node ids for every named pattern binding candidate."""
    return tuple(candidate.node_id for candidate in pattern_binder_candidates(pattern))


# ---------------------------------------------------------------------------
# Binder nodes (block-item level, not independently usable as Expr)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LetDecl:
    """``let pattern [: type] = expr`` — immutable binding (scopes over continuation).

    Pattern-bearing lets are represented uniformly in the AST. Scope and
    typechecking resolve supported patterns and record their selected bindings;
    the AST itself makes no claim about later-stage execution support. ``node_id``
    identifies the let match site, not any individual binder; binder identities
    come from the pattern nodes.

    ``scope_path`` is non-empty only for the ``let A::x = expr`` shorthand, which
    the parser reinterprets from a root-position bare qualifier chain pattern;
    ``pattern`` is then a plain ``VarPattern`` for the chain's member name.
    """

    pattern: Pattern
    type_ann: TypeExpr | None
    value: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
    scope_path: tuple[ScopeSegment, ...] = ()


def simple_let_pattern_name(pattern: Pattern) -> str | None:
    """Return a simple let root's name, with ``_`` for a wildcard root."""
    if isinstance(pattern, VarPattern):
        return pattern.name
    if isinstance(pattern, WildcardPattern):
        return "_"
    return None


@dataclass(frozen=True, slots=True)
class VarDecl:
    """``var [scope_path::]name [: type] = expr`` — mutable binding (scopes over continuation).

    ``scope_path`` is non-empty for the ``var A::count = expr`` shorthand.
    """

    name: str
    type_ann: TypeExpr | None
    value: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
    scope_path: tuple[ScopeSegment, ...] = ()


@dataclass(frozen=True, slots=True)
class NameTarget:
    """Assignment target for ``name := expr``.

    ``qualifier`` is set for a qualified assignment target such as
    ``std/config::max-iters := expr``; it is ``None`` for a plain local target.
    """

    name: str
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
    qualifier: QualifierChain | None = None


@dataclass(frozen=True, slots=True)
class IndexTarget:
    """Assignment target for ``root[index] := expr``."""

    obj: Expr
    index: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


AssignTarget = NameTarget | IndexTarget


@dataclass(frozen=True, slots=True)
class AssignStmt:
    """``target := expr`` — assignment to a mutable target.  Yields ``unit``."""

    target: AssignTarget
    value: Expr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


# Closed union of binder nodes.
Binder = LetDecl | VarDecl | AssignStmt


# ---------------------------------------------------------------------------
# Declaration nodes (top-level + block-level constructs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecordDef(GenericDeclaration):
    """``record Name(fields)`` declaration."""

    name: str
    fields: tuple[Param, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
    type_param_slots: tuple[str, ...] = ()
    is_builtin: bool = False
    scope_path: tuple[ScopeSegment, ...] = ()


@dataclass(frozen=True, slots=True)
class VariantDef:
    """A single variant inside an ``enum`` declaration."""

    name: str
    fields: tuple[Param, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class EnumDef(GenericDeclaration):
    """``enum Name { variants }`` declaration."""

    name: str
    variants: tuple[VariantDef, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
    type_param_slots: tuple[str, ...] = ()
    is_builtin: bool = False
    scope_path: tuple[ScopeSegment, ...] = ()


@dataclass(frozen=True, slots=True)
class ExceptionDef(GenericDeclaration):
    """``exception Name [extends Base](fields...)`` declaration."""

    name: str
    fields: tuple[Param, ...]
    base: str | None
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
    type_param_slots: tuple[str, ...] = ()
    is_builtin: bool = False
    scope_path: tuple[ScopeSegment, ...] = ()


@dataclass(frozen=True, slots=True)
class TypeAlias(GenericDeclaration):
    """``type Name = TypeExpr`` declaration."""

    name: str
    type_expr: TypeExpr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
    type_param_slots: tuple[str, ...] = ()
    scope_path: tuple[ScopeSegment, ...] = ()


@dataclass(frozen=True, slots=True)
class ParamDecl:
    """``param name[: TypeExpr] [= expr]`` declaration.

    The type ``annotation`` is optional: ``param spec`` is equivalent to
    ``param spec: text``.  The default (``text``) is applied by the TYPECHECK
    pass, not synthesized by the parser, so ``annotation`` is ``None`` when the
    source omits it.

    The ``default`` expression is optional; ``None`` when omitted.

    ``scope_path`` is non-empty for a ``param`` declared as a member of a named
    scope region; there is no declaration-path shorthand for ``param``, so this
    is always the enclosing region's path, never a prefix parsed from the
    declaration head itself.
    """

    name: str
    annotation: TypeExpr | None
    default: Expr | None
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
    scope_path: tuple[ScopeSegment, ...] = ()


def scoped_public_name(scope_path: tuple[ScopeSegment, ...], name: str) -> str:
    """Return the full path spelling of a scoped binding's or param's public name.

    A root declaration's public name is its bare name; a scoped one's is its
    full path spelling (``"Deploy::region"``), matching how the language
    itself addresses the member from outside its scope. This is display
    text, not an identity key — callers that need the scope path keep it
    structured rather than recovering it by splitting this spelling.
    """
    from agm.agl.modules.ids import spell_scope_path

    return spell_scope_path((*(segment.name for segment in scope_path), name))


def param_external_key(param: ParamDecl) -> str:
    """Return *param*'s external key: the CLI flag and config-table spelling.

    Shares its spelling with the public name of a scoped ``let``/``var``
    binding; see ``scoped_public_name``.
    """
    return scoped_public_name(param.scope_path, param.name)


@dataclass(frozen=True, slots=True)
class ProgramDecl:
    """``program NAME`` declaration used for host config lookup."""

    name: str
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class AgentDecl:
    """``agent NAME [= "runner string"]`` declaration.

    ``runner`` is the optional static runner-command hint (a literal string
    with NO interpolation); ``None`` for a bare declaration.
    In AgL, agent names are ordinary value bindings of type ``agent``.
    """

    name: str
    runner: str | None
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
    scope_path: tuple[ScopeSegment, ...] = ()


@dataclass(frozen=True, slots=True)
class BuiltinVarDecl:
    """``builtin var NAME : Type`` declaration — a body-less, runtime-backed,
    MUTABLE binding.

    Mirrors ``builtin def`` / ``builtin record`` (a host-provided declaration with
    a signature but no body).  A ``builtin var`` names an engine setting whose
    value lives in an interpreter register: programs read it as an ordinary value
    and assign it with ``:=``.  The declaration itself introduces no initializer
    and lowers to nothing.

    ``name``      — the declared engine key (kebab-case, e.g. ``"max-iters"``).
    ``type_ann``  — the mandatory declared type (no value expression).

    ``scope_path`` is non-empty when the declaration sits inside a named scope
    region; it has no declaration-path shorthand of its own.
    """

    name: str
    type_ann: TypeExpr
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
    scope_path: tuple[ScopeSegment, ...] = ()


@dataclass(frozen=True, slots=True)
class InfixDecl:
    """``infixl OP [at priority]`` or ``infixr OP [at priority]`` declaration."""

    name: str
    assoc: InfixAssoc
    priority: int | None
    priority_base: str | None
    priority_delta: int
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


@dataclass(frozen=True, slots=True)
class ScopeRegion:
    """A named scope region containing static declarations or nested regions.

    A multi-segment source header is normalized into nested single-segment
    regions, so each node owns exactly one :class:`ScopeSegment`.
    """

    segment: ScopeSegment
    items: tuple[ScopeItem, ...]
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)


ScopedDeclaration = FuncDef | RecordDef | EnumDef | ExceptionDef | TypeAlias | AgentDecl


def is_scoped_declaration(node: object) -> TypeGuard[ScopedDeclaration]:
    """Whether *node* is a declaration owned by a non-root named scope."""
    return isinstance(
        node, (FuncDef, RecordDef, EnumDef, ExceptionDef, TypeAlias, AgentDecl)
    ) and bool(node.scope_path)


def static_items(items: tuple[Item, ...]) -> Iterator[Item]:
    """Yield block items, descending into named scope regions.

    Named scope regions are transparent to whole-module item collection: a
    region's own items are spliced into its parent's stream, in textual
    order, so a caller that needs every item regardless of nesting depth —
    e.g. discovering every ``param`` declaration for its external key — walks
    one flat sequence. ``static_type_items`` and ``static_function_items``
    are this walk narrowed to one item kind.
    """
    for item in items:
        if isinstance(item, ScopeRegion):
            yield from static_items(item.items)
        else:
            yield item


def static_type_items(
    items: tuple[Item, ...],
) -> Iterator[RecordDef | EnumDef | ExceptionDef | TypeAlias]:
    """Yield type declarations, descending into named scope regions.

    Named scope regions are transparent to declaration collection: a scoped
    declaration carries its own structured scope path, so passes that walk
    static declarations see a region's members as siblings of its own items.
    """
    for item in static_items(items):
        if isinstance(item, (RecordDef, EnumDef, ExceptionDef, TypeAlias)):
            yield item


def static_function_items(items: tuple[Item, ...]) -> Iterator[FuncDef]:
    """Yield function declarations, descending into named scope regions."""
    for item in static_items(items):
        if isinstance(item, FuncDef):
            yield item


# Closed union of declaration nodes.
# FuncDef is a declaration (top-level or block-level named function).
Declaration = (
    FuncDef
    | RecordDef
    | EnumDef
    | ExceptionDef
    | TypeAlias
    | ParamDecl
    | ProgramDecl
    | AgentDecl
    | BuiltinVarDecl
    | InfixDecl
    | ImportDecl
    | ExportDecl
    | OpenDecl
)


# ---------------------------------------------------------------------------
# Item unions
# ---------------------------------------------------------------------------

# Scope-region items are static declarations, value bindings, module-system
# declarations, or nested regions. The parser enforces this restricted subset
# before it crosses the AST firewall.
ScopeItem = (
    ScopeRegion
    | OpenDecl
    | ImportDecl
    | ExportDecl
    | FuncDef
    | RecordDef
    | EnumDef
    | ExceptionDef
    | TypeAlias
    | AgentDecl
    | LetDecl
    | VarDecl
    | ParamDecl
    | BuiltinVarDecl
)

# An item is anything that can appear in a block sequence:
# declarations (introduce names), binders (scope over the rest), expressions,
# or a named scope region.
Item = Declaration | Binder | Expr | ScopeRegion


# ---------------------------------------------------------------------------
# Program root
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Program:
    """Root node of an AgL program.  ``body`` is a ``Block`` of top-level items."""

    body: Block
    span: SourceSpan = dc_field(compare=False)
    node_id: int = dc_field(compare=False)
