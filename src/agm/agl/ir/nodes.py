"""IR node types for the AgL typeless execution IR.

Every node is a frozen dataclass with a ``location: Location`` field.
Child collections are ``tuple`` (never ``list``).

``IrExpr`` is the closed union of all expression node types defined here.
The evaluator and lowerer dispatch over it with a structural ``match`` whose
final arm is ``assert_never(node)``, so mypy exhaustiveness makes a
missing case a compile-time error.

``IrIndexStep`` is a helper child record used by ``IrAssign``; it is NOT a
member of ``IrExpr``.

Invariant: ``IrSequence`` and ``IrBlock``
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
from agm.agl.ir.ids import ContractId, FunctionId, Location, NominalId, SymbolId
from agm.agl.ir.operations import (
    ArithKind,
    ArithOp,
    CmpOp,
    Coercion,
    CompareKind,
    ContainsKind,
    IndexKind,
    IterKind,
    NumericKind,
    UnaryOp,
)

__all__ = [
    "AutoTraceField",
    "IrAgentHandle",
    "IrAnd",
    "IrArith",
    "IrAsk",
    "IrAskRequest",
    "IrAssign",
    "IrExec",
    "IrBind",
    "IrBindPlan",
    "IrBlock",
    "IrBreak",
    "IrCapture",
    "IrCase",
    "IrCaseArm",
    "IrCatchHandler",
    "IrCoerce",
    "IrCompare",
    "IrConfigBind",
    "IrConstBool",
    "IrConstDecimal",
    "IrConstInt",
    "IrConstJsonNull",
    "IrConstText",
    "IrConstUnit",
    "IrConstructorPlan",
    "IrContains",
    "IrContinue",
    "IrConvert",
    "IrIterHasNext",
    "IrIterInit",
    "IrIterNext",
    "IrDirectCall",
    "IrExpr",
    "IrField",
    "IrFunctionParam",
    "IrIf",
    "IrIfBranch",
    "IrIndex",
    "IrIndexStep",
    "IrIndirectCall",
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
    "IrParseJson",
    "IrPrint",
    "IrRaise",
    "IrRenderValue",
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
    For ``var`` symbols this reads through the cell.
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
    lowering time; the evaluator switches on it without runtime type sniffing.

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

    Invariant: ``len(items) >= 1`` (enforced by ``validate_ir``).

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

    Invariant: ``len(items) >= 1`` (enforced by ``validate_ir``).

    Distinguished from ``IrSequence``: ``IrBlock`` corresponds one-to-one
    with a ``Block`` node in the source AST.  ``IrSequence`` is used for
    lowering-internal sequencing with no direct source counterpart.
    """

    location: Location
    items: "tuple[IrExpr, ...]"


# ---------------------------------------------------------------------------
# Operator nodes
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
# Field/index access and template nodes
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
# Constructor nodes
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
    """IR enum-variant construction: ``EnumName::Variant(field = expr, ...)``.

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
    operand's enum type).  ``nominal`` records the tested enum for
    completeness and validation.
    """

    location: Location
    nominal: NominalId
    variant: str
    value: "IrExpr"
    negated: bool


# ---------------------------------------------------------------------------
# Control-flow nodes
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
# Match plan nodes — closed tagged-data union for case patterns
# ---------------------------------------------------------------------------
# These are NOT members of IrExpr; they appear only as IrCaseArm.plan.
# Dispatch over IrMatchPlan uses a closed match/assert_never.


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
#: Dispatch with a structural ``match`` / ``assert_never``.
IrMatchPlan = IrWildcardPlan | IrBindPlan | IrLiteralPlan | IrVariantPlan | IrConstructorPlan


# ---------------------------------------------------------------------------
# Case expression helper and node
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
    frame before the arm body is evaluated.
    """

    location: Location
    subject: "IrExpr"
    arms: "tuple[IrCaseArm, ...]"


@dataclass(frozen=True, slots=True)
class IrLoop:
    """Unconditional repeat of ``body``.

    Repeats ``body`` forever.  The only exits are ``IrBreak`` (leave the loop,
    yielding ``UnitValue``) and ``IrContinue`` (start the next iteration).
    All richer loop features (``for``/``while``/``until``/``[n]`` bound) are
    **desugared into** ``body`` by the lowerer.

    There are NO per-iteration frames: body bindings reuse the same single
    frame slots across iterations.

    ``guarded`` marks loops that carry their own termination bound — a ``[n]``
    bound (which raises ``MaxIterationsExceeded`` itself) or a ``for`` clause
    (bounded by a finite collection).  The host's global ``max-iters`` safety
    valve applies ONLY to unguarded loops (``guarded=False``): a ``for`` over a
    million-element list or a ``do[n]`` with a large ``n`` must never be cut
    short by the host safety net, which exists solely to catch runaway
    unbounded ``while``/``do…until`` loops.
    """

    location: Location
    body: "IrExpr"
    guarded: bool = False


@dataclass(frozen=True, slots=True)
class IrBreak:
    """Exit the nearest enclosing ``IrLoop``, yielding ``UnitValue``.

    Implemented by raising an internal ``_BreakSignal`` Python exception in the
    evaluator; the signal propagates through ``IrTry`` bodies (which catch only
    ``AglRaise``) to the enclosing ``IrLoop`` handler.
    """

    location: Location


@dataclass(frozen=True, slots=True)
class IrContinue:
    """Proceed to the next iteration of the nearest enclosing ``IrLoop``.

    Implemented by raising an internal ``_ContinueSignal`` Python exception in
    the evaluator; the signal propagates through ``IrTry`` bodies to the
    enclosing ``IrLoop`` handler where it is caught and used to ``continue``
    the Python ``while True`` loop.
    """

    location: Location


@dataclass(frozen=True, slots=True)
class IrIterInit:
    """Initialize a loop iterator over a collection.

    ``kind`` selects list / dict-keys / text iteration.
    ``collection`` evaluates to the collection to iterate.
    Yields an ``IteratorValue`` (internal; never user-visible).
    """

    location: Location
    kind: "IterKind"
    collection: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrIterHasNext:
    """Test whether a loop iterator has more elements.

    ``iterator`` evaluates to an ``IteratorValue``.
    Yields ``BoolValue(True)`` when more elements remain.
    """

    location: Location
    iterator: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrIterNext:
    """Advance a loop iterator and return the current element.

    ``iterator`` evaluates to an ``IteratorValue``.
    Advances the iterator's position and returns the element at the
    previous position.  Caller must check ``IrIterHasNext`` first.
    """

    location: Location
    iterator: "IrExpr"


# ---------------------------------------------------------------------------
# Function/closure nodes
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


@dataclass(frozen=True, slots=True)
class IrIndirectCall:
    """IR indirect call to a function value (closure/lambda/first-class function).

    Used when the callee is an arbitrary expression (not a bare VarRef resolving to a
    function_binding).  Arguments are positional-only and are NOT coerced (the caller
    passes the raw evaluated argument; coercion at the call site is the direct-call
    responsibility only).  The result IS coerced by the FunctionDescriptor body
    (lower_coerced bakes the coercion into the body at lowering time).

    Depth-limit check happens AFTER the callee is evaluated, BEFORE arguments are
    bound — matching the legacy ``_apply_closure`` order.
    """

    location: Location
    callee: "IrExpr"
    arguments: "tuple[IrExpr, ...]"


# ---------------------------------------------------------------------------
# Host operation nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IrPrint:
    """IR host-op: ``print(value)`` — render *value* and write a line to stdout.

    Evaluates ``value``, renders it with the default single-line, unquoted
    options, then prints the rendered string. Returns the non-printable unit
    value.
    """

    location: Location
    value: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrRenderValue:
    """IR host-op: ``render(value)`` — render *value* to a text value."""

    location: Location
    value: "IrExpr"
    pretty: "IrExpr | None" = None
    quote_strings: "IrExpr | None" = None


@dataclass(frozen=True, slots=True)
class IrParseJson:
    """IR host-op: ``parse_json(text)`` — parse a JSON text value strictly.

    Evaluates ``value`` (always a ``TextValue`` in well-lowered IR), then calls
    ``parse_json_strict``.  On success returns ``JsonValue(obj)``; on
    ``StrictJsonParseError`` raises ``AglRaise`` with a ``JsonParseError``
    exception with the language-defined diagnostic fields.
    """

    location: Location
    value: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrAgentHandle:
    """IR host-op: evaluate to an AgentValue for the named agent.

    Emitted for AgentDecl lowering (agents are entry-only, bound once).
    Evaluates to ``AgentValue(name=agent_name)``.
    """

    location: Location
    agent_name: str


@dataclass(frozen=True, slots=True)
class IrAsk:
    """IR host-op: ask(prompt, agent:, on_parse_error:) builtin call.

    Evaluates ``agent`` (an AgentValue), ``prompt`` (text), dispatches to the
    registry, parses the response via the contract, and returns the typed Value.

    ``max_attempts``  — 1 for Abort/absent, 1+n for Retry(n).
    """

    location: Location
    agent: "IrExpr"
    prompt: "IrExpr"
    contract_id: "ContractId"
    max_attempts: int


@dataclass(frozen=True, slots=True)
class IrAskRequest:
    """IR host-op: ask-request(prompt, agent:) builtin call.

    Builds the AgentRequest record value WITHOUT dispatching the agent.
    Side-effect-free.
    """

    location: Location
    agent: "IrExpr"
    prompt: "IrExpr"
    contract_id: "ContractId"
    max_attempts: int


@dataclass(frozen=True, slots=True)
class IrExec:
    """IR host-op: exec(command, ...) builtin call."""

    location: Location
    command: "IrExpr"
    contract_id: "ContractId"
    max_attempts: int


# ---------------------------------------------------------------------------
# Config binding node (config/param unification — Task 3a)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IrConfigBind:
    """IR config-binding: runtime resolution of one engine config key.

    Evaluates in declaration order as a module-body initializer (NOT hoisted
    like ``IrParam``).  The evaluator resolves the binding value per the config
    precedence chain:

        CLI --X  >  source value (if ``value`` is not None)  >  config_base[X]

    then binds ``symbol`` to the resolved value; the binding is immutable per
    the scope pass.

    ``symbol``      — the linker-allocated SymbolId for this config binding.
    ``public_name`` — the kebab-case engine key (e.g. "max-iters").
    ``value``       — lowered source expression; ``None`` for bare ``config X``.
    """

    location: Location
    symbol: SymbolId
    public_name: str
    value: "IrExpr | None"


# ---------------------------------------------------------------------------
# Closed IrExpr union
# ---------------------------------------------------------------------------

#: Closed union of all IR expression node types defined in this module.
#: Closed union of all expression node types.
#:
#: Dispatch with a structural ``match`` whose final arm is
#: ``assert_never(node)`` so mypy exhaustiveness makes a missing case a
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
    | IrBreak
    | IrContinue
    | IrIterInit
    | IrIterHasNext
    | IrIterNext
    | IrMakeClosure
    | IrDirectCall
    | IrIndirectCall
    | IrPrint
    | IrRenderValue
    | IrParseJson
    | IrAgentHandle
    | IrAsk
    | IrAskRequest
    | IrExec
    | IrConfigBind
)
