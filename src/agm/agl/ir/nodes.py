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

from agm.agl.ir.contracts import ConversionFailureMode, ConversionRecipe
from agm.agl.ir.ids import FunctionId, Location, NominalId, SymbolId
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
    "AutoTraceField",
    "IrAnd",
    "IrArith",
    "IrAssign",
    "IrBind",
    "IrBindPlan",
    "IrBlock",
    "IrCapture",
    "IrCase",
    "IrCaseArm",
    "IrCatchHandler",
    "IrCoerce",
    "IrCompare",
    "IrConstBool",
    "IrConstDecimal",
    "IrConstInt",
    "IrConstJsonNull",
    "IrConstText",
    "IrConstUnit",
    "IrDirectCall",
    "IrConstructorPlan",
    "IrContains",
    "IrConvert",
    "IrExpr",
    "IrField",
    "IrFunctionParam",
    "IrIf",
    "IrIfBranch",
    "IrIndex",
    "IrIndexStep",
    "IrLiteralPlan",
    "IrLoad",
    "IrLoop",
    "IrMakeConstructor",
    "IrMakeClosure",
    "IrMakeDict",
    "IrMakeEnum",
    "IrMakeException",
    "IrMakeList",
    "IrMakeRecord",
    "IrMatchPlan",
    "IrOr",
    "IrRaise",
    "IrRenderTemplate",
    "IrSequence",
    "IrTemplateSegment",
    "IrTemplateText",
    "IrTemplateValue",
    "IrTry",
    "IrUnary",
    "IrVariantIs",
    "IrVariantPlan",
    "IrWildcardPlan",
    "UseDefault",
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
# Constructor nodes (M3d)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AutoTraceField:
    """Sentinel marker for an auto-injected trace_id field in IrMakeException.

    Each slot in ``IrMakeException.fields`` that was NOT provided by the caller
    carries this sentinel rather than an ``IrExpr``.  The evaluator allocates
    ONE ``TextValue(trace.new_event_id())`` per construction and substitutes it
    for every ``AutoTraceField`` slot in that construction.

    This is NOT an ``IrExpr`` member — it cannot appear in any other IR position.
    """


@dataclass(frozen=True, slots=True)
class IrMakeRecord:
    """IR record construction: ``RecordName(field: expr, ...)``.

    ``nominal`` — the ``NominalId`` of the record type.
    ``display_name`` — user-facing type name.
    ``fields`` — declaration-order tuple of ``(field_name, expr)`` pairs;
        each ``expr`` is already coerced to the declared field type by the
        lowerer via ``lower_coerced``.
    """

    location: Location
    nominal: NominalId
    display_name: str
    fields: "tuple[tuple[str, IrExpr], ...]"


@dataclass(frozen=True, slots=True)
class IrMakeEnum:
    """IR enum-variant construction: ``EnumName.Variant(field: expr, ...)``.

    ``nominal`` — the ``NominalId`` of the owning enum type.
    ``display_name`` — user-facing enum type name.
    ``variant`` — the variant name.
    ``fields`` — declaration-order tuple of ``(field_name, expr)`` pairs;
        each ``expr`` is already coerced by the lowerer.
    """

    location: Location
    nominal: NominalId
    display_name: str
    variant: str
    fields: "tuple[tuple[str, IrExpr], ...]"


@dataclass(frozen=True, slots=True)
class IrMakeException:
    """IR exception construction: ``ExcName(field: expr, ...)``.

    ``nominal`` — the ``NominalId`` of the exception type (uses PRELUDE_ID).
    ``display_name`` — user-facing exception type name.
    ``fields`` — declaration-order tuple of ``(field_name, slot)`` where
        ``slot`` is either a coerced ``IrExpr`` (explicitly provided by the
        caller) or an ``AutoTraceField`` sentinel (declared but not provided —
        will receive the construction's freshly allocated trace id).
    """

    location: Location
    nominal: NominalId
    display_name: str
    fields: "tuple[tuple[str, IrExpr | AutoTraceField], ...]"


@dataclass(frozen=True, slots=True)
class IrMakeConstructor:
    """IR first-class constructor reference.

    Evaluates to a ``ConstructorValue(nominal, display_name, variant)`` without
    constructing the record/enum.  Used when a constructor is referenced as a
    value (non-call position).

    ``variant`` is ``None`` for a record constructor; non-``None`` for an enum
    variant constructor.
    """

    location: Location
    nominal: NominalId
    display_name: str
    variant: "str | None"


@dataclass(frozen=True, slots=True)
class IrConvert:
    """IR cast / conversion (``as`` and fallible ``as?``).

    Evaluates ``value`` once, then runs ``recipe`` (a typeless
    ``ConversionRecipe``).  ``failure_mode`` selects behavior on a fallible
    failure: ``RAISE_CAST_ERROR`` raises a ``CastError`` (the ``as`` operator);
    ``RETURN_BOOL`` yields ``False`` (the fallible ``as?`` operator).  Total
    ``as?`` is lowered to ``IrSequence((source, IrConstBool(True)))`` and never
    reaches this node.
    """

    location: Location
    value: "IrExpr"
    recipe: ConversionRecipe
    failure_mode: ConversionFailureMode


@dataclass(frozen=True, slots=True)
class IrVariantIs:
    """IR enum-variant membership test (``is`` / ``is not``).

    Evaluates ``value`` (always an ``EnumValue`` in well-lowered IR) and yields
    ``BoolValue((value.variant == variant) != negated)``.  The boolean depends
    only on the variant string and ``negated`` — matching the legacy
    interpreter, which does not compare the nominal (the checker guarantees the
    operand's enum type).  ``nominal`` records the tested enum (D2) for
    completeness and validation.
    """

    location: Location
    nominal: NominalId
    variant: str
    value: "IrExpr"
    negated: bool


# ---------------------------------------------------------------------------
# Control-flow nodes (M3f-A)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IrIfBranch:
    """A single branch in an ``IrIf`` node.

    ``cond`` is ``None`` for the else branch (always taken), or an ``IrExpr``
    that must evaluate to ``BoolValue``.  ``body`` is evaluated when the branch
    is taken.

    This is NOT a member of ``IrExpr``.
    """

    cond: "IrExpr | None"
    body: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrIf:
    """IR if expression: evaluates branches in order, takes first matching one.

    ``has_else`` is ``True`` when the original source ``if`` had an ``else``
    branch (i.e. there is a branch with ``cond=None``).  The evaluator returns
    the taken branch's body value when ``has_else`` is ``True``, and
    ``UnitValue`` when ``has_else`` is ``False`` (regardless of which branch
    was taken) — matching legacy ``_eval_if`` value semantics.
    """

    location: Location
    branches: "tuple[IrIfBranch, ...]"
    has_else: bool


@dataclass(frozen=True, slots=True)
class IrRaise:
    """IR raise: evaluate ``exc`` (must yield ``ExceptionValue``) and propagate it."""

    location: Location
    exc: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrCatchHandler:
    """A single catch handler in an ``IrTry`` node.

    ``nominal`` and ``display_name`` together identify the exception type:
    - ``nominal=None, display_name=None`` — catch-all (catches everything).
    - ``nominal`` set, ``display_name`` set — specific match by ``display_name``
      string equality against ``ExceptionValue.display_name`` (legacy semantics).

    ``symbol`` is the ``SymbolId`` of the binding variable when the handler
    declares one (``catch SomeError e => ...``); ``None`` otherwise.  The
    evaluator writes the caught ``ExceptionValue`` into the frame under this
    symbol before evaluating ``body``.

    This is NOT a member of ``IrExpr``.
    """

    nominal: NominalId | None
    display_name: str | None
    symbol: SymbolId | None
    body: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrTry:
    """IR try/catch: evaluate ``body``; on ``AglRaise``, match handlers in order.

    Semantics mirror legacy ``_eval_try``:
    - Evaluate ``body``; if it completes normally, return its value.
    - On ``AglRaise``, iterate ``handlers`` in order; the first handler that
      matches (catch-all when ``display_name is None``; specific when
      ``display_name == exc.display_name``) wins.
    - If a handler matches and ``handler.symbol`` is not ``None``, bind the
      caught ``ExceptionValue`` in the current frame under that symbol.
    - Evaluate the handler's ``body`` and return its value.
    - If no handler matches, re-raise the original ``AglRaise`` unchanged.
    """

    location: Location
    body: "IrExpr"
    handlers: "tuple[IrCatchHandler, ...]"


# ---------------------------------------------------------------------------
# Match plan nodes (M3f-B) — closed tagged-data union for case patterns
# ---------------------------------------------------------------------------
# These are NOT members of IrExpr; they appear only as IrCaseArm.plan.
# Dispatch over IrMatchPlan uses a closed match/assert_never (D4).


@dataclass(frozen=True, slots=True)
class IrWildcardPlan:
    """Match-plan for the ``_`` wildcard pattern — always matches, no binding."""


@dataclass(frozen=True, slots=True)
class IrBindPlan:
    """Match-plan for a ``VarPattern`` binder — always matches, binds ``symbol``."""

    symbol: SymbolId


@dataclass(frozen=True, slots=True)
class IrLiteralPlan:
    """Match-plan for a ``LiteralPattern`` — matches when ``value_eq(subject, value)``."""

    value: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrVariantPlan:
    """Match-plan for a nullary bare-variant ``VarPattern`` — matches ``EnumValue.variant``."""

    variant: str


@dataclass(frozen=True, slots=True)
class IrConstructorPlan:
    """Match-plan for a ``ConstructorPattern`` — checks variant then recurses over fields."""

    variant: str
    fields: "tuple[tuple[str, IrMatchPlan], ...]"


#: Closed union of all match-plan node types.
#: Dispatch with a structural ``match`` / ``assert_never`` (D4).
IrMatchPlan = IrWildcardPlan | IrBindPlan | IrLiteralPlan | IrVariantPlan | IrConstructorPlan


# ---------------------------------------------------------------------------
# Case expression helper and node (M3f-B)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IrCaseArm:
    """A single arm in an ``IrCase`` node — NOT a member of ``IrExpr``.

    ``plan`` is the closed ``IrMatchPlan`` compiled from the source pattern.
    ``body`` is evaluated when the plan matches the subject.
    """

    plan: "IrMatchPlan"
    body: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrCase:
    """IR case expression: eval ``subject`` once; try each arm in order.

    The value of an ``IrCase`` is the body of the first arm whose plan
    matches the subject.  If no arm matches, raises ``AglRaise`` with a
    ``MatchError`` exception (built via ``make_match_error`` from
    ``agm.agl.eval.matching``).

    Semantics mirror legacy ``_eval_case`` first-match ordering exactly.
    Pattern binders (``IrBindPlan``) write their bindings into the current
    frame (D5) before the arm body is evaluated.
    """

    location: Location
    subject: "IrExpr"
    arms: "tuple[IrCaseArm, ...]"


@dataclass(frozen=True, slots=True)
class IrLoop:
    """IR do…until loop (M3f-C).

    Evaluates ``body`` then ``condition`` up to ``limit`` times.  When
    ``condition`` evaluates to ``BoolValue(True)`` the loop exits and yields
    ``UnitValue``.  When ``limit`` iterations elapse without the condition
    becoming ``True``, raises ``AglRaise`` with a ``MaxIterationsExceeded``
    exception whose fields mirror the legacy interpreter exactly.

    ``limit=None`` means "use the evaluator's configured default loop limit"
    (mirrors ``Do.limit is None`` → ``self._loop_limit`` in the legacy
    interpreter).  Do NOT bake the default into the IR — it is a runtime /
    configuration concern.

    ``condition_source`` is the pre-sliced source-text of the condition
    expression (mirrors ``_source_slice(expr.condition.span)`` in the legacy
    interpreter).  The IR evaluator has no AST spans, so the lowerer captures
    this string and embeds it here for use in the ``MaxIterationsExceeded``
    ``condition`` field.

    Per D5 there are NO per-iteration frames: body bindings reuse the same
    single frame slots across iterations, matching legacy observable behaviour
    (only the ``until`` condition reads body-bound vars).
    """

    location: Location
    limit: "int | None"
    body: "IrExpr"
    condition: "IrExpr"
    condition_source: str


# ---------------------------------------------------------------------------
# Function/closure nodes (M4a)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IrCapture:
    """A captured outer variable in an IrMakeClosure.

    by_cell: True for var (share the Cell), False for let/param (snapshot value).
    """

    symbol: SymbolId
    by_cell: bool


@dataclass(frozen=True, slots=True)
class UseDefault:
    """Sentinel in IrDirectCall.arguments: use the param default for this arg."""

    param_index: int


@dataclass(frozen=True, slots=True)
class IrFunctionParam:
    """A function parameter in a FunctionDescriptor."""

    symbol: SymbolId
    default: "IrExpr | None"


@dataclass(frozen=True, slots=True)
class IrMakeClosure:
    """IR closure creation: evaluates to an IrClosureValue."""

    location: Location
    function_id: FunctionId
    captures: "tuple[IrCapture, ...]"


@dataclass(frozen=True, slots=True)
class IrDirectCall:
    """IR direct call to a named user function."""

    location: Location
    function_id: FunctionId
    arguments: "tuple[IrExpr | UseDefault, ...]"


# ---------------------------------------------------------------------------
# Closed IrExpr union
# ---------------------------------------------------------------------------

#: Closed union of all IR expression node types defined in this module.
#: Grows in M4 as function/call families are wired in.
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
    | IrMakeRecord
    | IrMakeEnum
    | IrMakeException
    | IrMakeConstructor
    | IrVariantIs
    | IrConvert
    | IrIf
    | IrRaise
    | IrTry
    | IrCase
    | IrLoop
    | IrMakeClosure
    | IrDirectCall
)
