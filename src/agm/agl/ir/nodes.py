"""IR node types for the AgL typeless execution IR — M1 skeleton subset.

Every node is a frozen dataclass with a ``location: Location`` field.
Child collections are ``tuple`` (never ``list``).

This module contains the M1 subset only.  Families deferred to later
milestones (left as comments so later agents know what to add):

  M3 — arithmetic/comparison/contains/and/or nodes (IrArith, IrCompare,
        IrContains, IrAnd, IrOr); constructors (IrMakeRecord, IrMakeVariant);
        field/index access (IrFieldAccess, IrIndexAccess); cast/is-test
        (IrCast, IrIsTest); unary (IrUnaryNeg, IrUnaryNot);
        template strings (IrTemplate); if/case/do/try/raise expressions;
        pattern-match plans (IrMatchPlan and friends).

        Forward warnings for M3:
        - ``IrArith(ADD, ...)`` currently pairs ``ArithOp.ADD`` only with
          ``NumericKind`` (INT/DECIMAL), but AgL allows ``text + text`` string
          concatenation — M3 must add a text/string representation for ADD.
        - ``IrCompare`` equality (EQ/NEQ) is broader than ordering: the
          checker's ``comparable_types`` permits bool/record/enum/list/dict
          equality, so M3 needs an equality kind wider than the current
          ``CompareKind`` (INT/DECIMAL/TEXT), which covers only ordering.
  M4 — functions/lambdas/closures (IrFuncDef, IrLambda, IrCall, IrTailCall,
        IrReturn); param binding (IrParam); free-var capture descriptors.
  M5 — cross-module symbol references; module init ordering.
  M6 — host-prep metadata (agent declarations, param descriptors).

``IrExpr`` is the closed union of all expression node types defined here.
The evaluator and lowerer dispatch over it with a structural ``match`` whose
final arm is ``assert_never(node)`` (D4), so mypy exhaustiveness makes a
missing case a compile-time error.

``IrIndexStep`` is a helper child record used by ``IrAssign``; it is NOT a
member of ``IrExpr``.

Invariant (enforced by the validator in M1-C): ``IrSequence`` and ``IrBlock``
must be non-empty (``len(items) >= 1``).  The validator checks this; do not
rely on the constructor to enforce it, so that the linker can build nodes
incrementally.

``IrBlock`` mirrors a source-level ``Block`` node (one-to-one with a curly-
brace sequence in the program source).  ``IrSequence`` is a lowering-internal
compound: the lowerer uses it to sequence an effectful sub-expression together
with its result (e.g. a side-effecting initializer followed by the load of the
fresh binding).
"""

from __future__ import annotations

import decimal
from dataclasses import dataclass

from agm.agl.ir.ids import Location, SymbolId
from agm.agl.ir.operations import (
    ArithKind,
    ArithOp,
    CmpOp,
    Coercion,
    CompareKind,
    ContainsKind,
    IndexKind,
    NumericKind,
    UnaryOp,
)

__all__ = [
    "IrAnd",
    "IrArith",
    "IrAssign",
    "IrBind",
    "IrBlock",
    "IrCoerce",
    "IrCompare",
    "IrConstBool",
    "IrConstDecimal",
    "IrConstInt",
    "IrConstJsonNull",
    "IrConstText",
    "IrConstUnit",
    "IrContains",
    "IrExpr",
    "IrField",
    "IrIndex",
    "IrIndexStep",
    "IrLoad",
    "IrMakeDict",
    "IrMakeList",
    "IrOr",
    "IrRenderTemplate",
    "IrSequence",
    "IrTemplateSegment",
    "IrTemplateText",
    "IrTemplateValue",
    "IrUnary",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IrConstInt:
    """IR constant: a fixed integer value."""

    location: Location
    value: int


@dataclass(frozen=True, slots=True)
class IrConstDecimal:
    """IR constant: a fixed decimal (fixed-point) value."""

    location: Location
    value: decimal.Decimal


@dataclass(frozen=True, slots=True)
class IrConstBool:
    """IR constant: a boolean value (``True`` or ``False``)."""

    location: Location
    value: bool


@dataclass(frozen=True, slots=True)
class IrConstText:
    """IR constant: a plain text (string) value."""

    location: Location
    value: str


@dataclass(frozen=True, slots=True)
class IrConstUnit:
    """IR constant: the unit value ``()``."""

    location: Location


@dataclass(frozen=True, slots=True)
class IrConstJsonNull:
    """IR constant: the JSON ``null`` value."""

    location: Location


# ---------------------------------------------------------------------------
# Container literals
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IrMakeList:
    """IR list construction: ``[items...]``.

    Each element is an ``IrExpr`` evaluated left-to-right.
    Mirrors the AST ``ListLit`` node.
    """

    location: Location
    items: "tuple[IrExpr, ...]"


@dataclass(frozen=True, slots=True)
class IrMakeDict:
    """IR dict construction: ``{k: v, ...}``.

    Each entry is a ``(key_expr, value_expr)`` pair evaluated left-to-right.
    Mirrors the AST ``DictLit`` node (whose ``DictEntry.key`` is a
    ``StringLit``; at IR level keys are already resolved to ``IrExpr``).
    """

    location: Location
    entries: "tuple[tuple[IrExpr, IrExpr], ...]"


# ---------------------------------------------------------------------------
# Bindings / storage
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IrLoad:
    """IR load: read the current value of a symbol.

    For ``let`` symbols this is the stored value directly.
    For ``var`` symbols this reads through the cell (D5).
    """

    location: Location
    symbol: SymbolId


@dataclass(frozen=True, slots=True)
class IrBind:
    """IR bind: introduce a new binding for ``symbol`` with the given ``value``.

    Corresponds to ``LetDecl`` (``let``) and ``VarDecl`` (``var``) at the
    lowered level.  Whether the binding is mutable is recorded in the
    ``SymbolDescriptor`` for ``symbol`` (``mutable`` field).
    """

    location: Location
    symbol: SymbolId
    value: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrIndexStep:
    """A single index step on a mutable assignment path.

    Used by ``IrAssign`` to represent ``target[index] := value`` paths.
    This is a helper child record and is NOT a member of ``IrExpr``.
    """

    kind: IndexKind
    index: "IrExpr"
    location: Location


@dataclass(frozen=True, slots=True)
class IrAssign:
    """IR assignment: ``symbol[path...] := value``.

    When ``path`` is empty this is a simple variable assignment (``x := v``).
    When ``path`` is non-empty it is a chained index assignment
    (``x[i][j] := v``).  The mutable root ``symbol`` must be a ``var``
    (``mutable=True`` in its ``SymbolDescriptor``).
    """

    location: Location
    symbol: SymbolId
    path: tuple[IrIndexStep, ...]
    value: "IrExpr"


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IrCoerce:
    """IR coercion: apply a resolved ``Coercion`` to ``value``.

    The ``operation`` field is a closed ``Coercion`` union member resolved at
    lowering time; the evaluator switches on it without runtime type sniffing
    (D3).

    The ``operation`` field is always a concrete ``Coercion`` — it never holds
    ``None``.  An identity (no-op) coercion is represented by the lowerer
    **omitting the ``IrCoerce`` node entirely** rather than emitting one with a
    null operation.  The ``Coercion | None`` shape belongs to the future
    ``compile_coercion`` helper's return type (where ``None`` signals "no node
    needed"), not to this field.
    """

    location: Location
    value: "IrExpr"
    operation: Coercion


# ---------------------------------------------------------------------------
# Sequencing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IrSequence:
    """IR lowering-internal compound: a sequence of expressions.

    The value of an ``IrSequence`` is the value of its last item.

    Invariant: ``len(items) >= 1`` (enforced by ``validate_ir`` in M1-C).

    Distinguished from ``IrBlock``: ``IrSequence`` is a lowering-internal
    construct used by the lowerer to sequence an effectful sub-expression
    together with its result (e.g. a side-effecting initializer followed by a
    load of the fresh binding).  It has no direct counterpart in the source.
    """

    location: Location
    items: "tuple[IrExpr, ...]"


@dataclass(frozen=True, slots=True)
class IrBlock:
    """IR block: mirrors a source-level ``Block`` (curly-brace sequence).

    The value of an ``IrBlock`` is the value of its last item.

    Invariant: ``len(items) >= 1`` (enforced by ``validate_ir`` in M1-C).

    Distinguished from ``IrSequence``: ``IrBlock`` corresponds one-to-one
    with a ``Block`` node in the source AST.  ``IrSequence`` is used for
    lowering-internal sequencing with no direct source counterpart.
    """

    location: Location
    items: "tuple[IrExpr, ...]"


# ---------------------------------------------------------------------------
# Operator nodes (M3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IrArith:
    """IR arithmetic: binary arithmetic operation (add, sub, mul, div)."""

    location: Location
    op: ArithOp
    kind: ArithKind
    lhs: "IrExpr"
    rhs: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrCompare:
    """IR comparison: binary comparison operation (eq, neq, lt, le, gt, ge)."""

    location: Location
    op: CmpOp
    kind: CompareKind
    lhs: "IrExpr"
    rhs: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrContains:
    """IR containment: x in container (list, dict, text)."""

    location: Location
    kind: ContainsKind
    item: "IrExpr"
    container: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrAnd:
    """IR short-circuit and."""

    location: Location
    lhs: "IrExpr"
    rhs: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrOr:
    """IR short-circuit or."""

    location: Location
    lhs: "IrExpr"
    rhs: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrUnary:
    """IR unary: NOT (logical) or NEG (numeric).

    ``kind`` is ``None`` for NOT, and a ``NumericKind`` for NEG.
    """

    location: Location
    op: UnaryOp
    kind: "NumericKind | None"
    value: "IrExpr"


# ---------------------------------------------------------------------------
# Field/index access and template nodes (M3c)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IrField:
    """IR field read: obj.field on a record or exception value."""

    location: Location
    value: "IrExpr"
    field: str


@dataclass(frozen=True, slots=True)
class IrIndex:
    """IR index access: obj[index] on a list (LIST) or dict (DICT)."""

    location: Location
    kind: IndexKind
    value: "IrExpr"
    index: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrTemplateText:
    """A literal text fragment in a template — NOT an IrExpr."""

    text: str


@dataclass(frozen=True, slots=True)
class IrTemplateValue:
    """An interpolated expression in a template — NOT an IrExpr."""

    value: "IrExpr"


#: Closed union of template segment types (not members of IrExpr).
IrTemplateSegment = IrTemplateText | IrTemplateValue


@dataclass(frozen=True, slots=True)
class IrRenderTemplate:
    """IR template rendering."""

    location: Location
    segments: "tuple[IrTemplateSegment, ...]"


# ---------------------------------------------------------------------------
# Closed IrExpr union
# ---------------------------------------------------------------------------

#: Closed union of all IR expression node types defined in this module (M1
#: subset).  Grows in M3/M4 as additional families are wired in.
#:
#: Dispatch with a structural ``match`` whose final arm is
#: ``assert_never(node)`` (D4) so mypy exhaustiveness makes a missing case a
#: compile-time error at ``just check``.
IrExpr = (
    IrConstInt
    | IrConstDecimal
    | IrConstBool
    | IrConstText
    | IrConstUnit
    | IrConstJsonNull
    | IrMakeList
    | IrMakeDict
    | IrLoad
    | IrBind
    | IrAssign
    | IrCoerce
    | IrSequence
    | IrBlock
    | IrArith
    | IrCompare
    | IrContains
    | IrAnd
    | IrOr
    | IrUnary
    | IrField
    | IrIndex
    | IrRenderTemplate
)
