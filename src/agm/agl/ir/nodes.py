"""IR node types for the AgL typeless execution IR.

Every node is a frozen dataclass with a ``location: Location`` field.
Child collections are ``tuple`` (never ``list``).

``IrExpr`` is the closed union of all expression node types defined here.
The evaluator and lowerer dispatch over it with a structural ``match`` whose
final arm is ``assert_never(node)``, so mypy exhaustiveness makes a
missing case a compile-time error.

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
import enum
from dataclasses import dataclass
from typing import TypeAlias

from agm.agl.ir.builtin_vars import BuiltinVarKey
from agm.agl.ir.contracts import ConversionFailureMode, ConversionRecipe
from agm.agl.ir.ids import ContractId, FunctionId, Location, NominalId, SymbolId
from agm.agl.ir.operations import (
    ArithKind,
    ArithOp,
    CmpOp,
    Coercion,
    CompareKind,
    ContainsKind,
    CopyKind,
    IndexKind,
    IterKind,
    NumericKind,
    UnaryOp,
)

__all__ = [
    "IrAnd",
    "IrArith",
    "IrAsk",
    "IrAskRequest",
    "IrSessionAsk",
    "IrSessionDefault",
    "IrSessionOp",
    "IrSessionOpen",
    "IrAssign",
    "IrExec",
    "IrBind",
    "IrBlock",
    "IrBreak",
    "IrBuiltinLoad",
    "IrBuiltinStore",
    "IrCapture",
    "IrCase",
    "IrCaseArm",
    "IrCaseKey",
    "IrCatchHandler",
    "IrCoerce",
    "IrCompare",
    "IrConstBool",
    "IrConstDecimal",
    "IrConstInt",
    "IrConstJsonNull",
    "IrConstText",
    "IrResource",
    "IrConstUnit",
    "IrContains",
    "IrContinue",
    "IrConvert",
    "IrCopyValue",
    "IrIterHasNext",
    "IrIterInit",
    "IrIterNext",
    "IrDirectCall",
    "IrExpr",
    "IrField",
    "IrFieldMode",
    "IrFunctionParam",
    "IrIf",
    "IrIfBranch",
    "IrIndex",
    "IrIndexSet",
    "IrIndirectCall",
    "IrNominalCaseKey",
    "IrLiteralCaseKey",
    "IrLiteralKind",
    "IrLiteralScalar",
    "IrLoad",
    "IrLoop",
    "IrMakeConstructor",
    "IrMakeClosure",
    "IrMakeDict",
    "IrMakeException",
    "IrMakeArray",
    "IrMakeJsonArray",
    "IrMakeJsonObject",
    "IrMakeRecord",
    "IrOr",
    "IrPrint",
    "IrRaise",
    "IrReturn",
    "IrRenderValue",
    "IrRenderTemplate",
    "IrSequence",
    "IrTemplateSegment",
    "IrTemplateText",
    "IrTemplateValue",
    "IrTry",
    "IrUnary",
    "IrUpdateRecord",
    "IrNominalCast",
    "IrNominalIs",
    "UseDefault",
    "is_canonical_literal_scalar",
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
class IrResource:
    """IR resource path resolved to an absolute filesystem location at link time."""

    location: Location
    path: str


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
class IrMakeArray:
    """IR array construction: ``[items...]``.

    Each element is an ``IrExpr`` evaluated left-to-right.
    Mirrors the AST ``ArrayLit`` node.
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


@dataclass(frozen=True, slots=True)
class IrMakeJsonArray:
    """IR direct JSON array construction: an ``array`` literal typed ``json``.

    Mirrors ``IrMakeArray`` in shape, but every item is already typed ``json``
    (a scalar coerced to ``json``, an already-``json`` expression, or a
    nested ``json``-typed literal), so evaluation builds a ``JsonValue``
    directly from the elements' JSON payloads. No AgL ``array`` value is ever
    materialized, and no ``IrCoerce`` is involved.
    """

    location: Location
    items: "tuple[IrExpr, ...]"


@dataclass(frozen=True, slots=True)
class IrMakeJsonObject:
    """IR direct JSON object construction: a ``dict`` literal typed ``json``.

    Mirrors ``IrMakeDict`` in shape and key rules (including the checker's
    duplicate-key rejection); every entry value is already typed ``json``.
    Evaluation builds a ``JsonValue`` directly from the elements' JSON
    payloads. No AgL ``dict`` value is ever materialized, and no ``IrCoerce``
    is involved.
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
class IrAssign:
    """IR assignment: ``symbol := value``, a simple ``var``-cell store.

    The mutable root ``symbol`` must be a ``var`` (``mutable=True`` in its
    ``SymbolDescriptor``).
    """

    location: Location
    symbol: SymbolId
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
    """IR containment: x in container (array, dict, text)."""

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


class IrFieldMode(enum.Enum):
    """Nominal identity rule for an ``IrField`` projection."""

    EXACT = "exact"
    UPPER_BOUND = "upper_bound"


@dataclass(frozen=True, slots=True)
class IrField:
    """IR nominal field projection from a record, enum payload, or exception.

    ``nominal`` identifies the field-bearing nominal shape used for static
    field validation. ``mode`` records whether runtime identity must match that
    nominal exactly or whether it is a static upper bound for the runtime
    nominal. Lowering chooses the mode because it knows whether the receiver
    was discriminated; the source-level declaration supplies the bound.
    Enum fields are validated against the union of their variant payload
    shapes. The default preserves exact checking for hand-built IR.
    """

    location: Location
    value: "IrExpr"
    nominal: NominalId
    field: str
    mode: IrFieldMode = IrFieldMode.EXACT


@dataclass(frozen=True, slots=True)
class IrUpdateRecord:
    """IR functional update: ``target with field = expr, ...``.

    Copies the target record or exception value with the listed fields
    replaced, preserving the runtime nominal (an update through a base-typed
    exception binding keeps the concrete exception type).

    ``updates`` — source-order tuple of ``(field_name, expr)`` pairs; each
        ``expr`` is already coerced to the declared field type by the lowerer
        via ``lower_coerced``.
    """

    location: Location
    value: "IrExpr"
    updates: "tuple[tuple[str, IrExpr], ...]"


@dataclass(frozen=True, slots=True)
class IrIndex:
    """IR index access: obj[index] on an array, dict, or text value."""

    location: Location
    kind: IndexKind
    value: "IrExpr"
    index: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrIndexSet:
    """IR indexed assignment: ``container[index] := value``.

    ``container`` evaluates to the array or dict being mutated; text is immutable and
    cannot produce this node. Under
    reference semantics it needs no root symbol or ``Cell`` — only a
    container reference, which ``container`` supplies directly. Nesting
    (``m["a"]["b"] := v``) falls out for free: ``container`` is itself an
    ``IrIndex`` that reads the inner container by reference. Mutates the
    container in place and evaluates to the non-printable unit, exactly as
    ``IrAssign`` does.
    """

    location: Location
    container: "IrExpr"
    kind: IndexKind
    index: "IrExpr"
    value: "IrExpr"


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
class IrMakeException:
    """IR exception construction: ``ExcName(field: expr, ...)``.

    ``nominal`` — the ``NominalId`` of the exception type: the shipped
        standard library's own reserved identity for a built-in exception a
        program declares nothing of its own for, or the declaring
        declaration's own identity otherwise — a program-declared
        ``builtin exception`` included.
    ``display_name`` — user-facing exception type name.
    ``fields`` — declaration-order tuple of ``(field_name, expr)`` pairs;
        each expression is coerced to the declared field type by the lowerer.
    """

    location: Location
    nominal: NominalId
    display_name: str
    fields: "tuple[tuple[str, IrExpr], ...]"


@dataclass(frozen=True, slots=True)
class IrMakeConstructor:
    """IR first-class constructor reference.

    Evaluates to a ``ConstructorValue(nominal, display_name)`` without
    constructing the record. Used when a constructor is referenced as a value
    (non-call position).
    """

    location: Location
    nominal: NominalId
    display_name: str


@dataclass(frozen=True, slots=True)
class IrConvert:
    """IR cast / conversion (``as`` and fallible ``as?``).

    Evaluates ``value`` once, then runs ``recipe`` (a typeless
    ``ConversionRecipe``).  ``failure_mode`` selects behavior on a fallible
    failure: ``RAISE_CAST_ERROR`` raises a ``CastError`` (the ``as`` operator);
    ``RETURN_BOOL`` makes ``as?`` evaluate to whether the conversion succeeded.
    """

    location: Location
    value: "IrExpr"
    recipe: ConversionRecipe
    failure_mode: ConversionFailureMode


@dataclass(frozen=True, slots=True)
class IrNominalCast:
    """Identity cast from an enum value to one of its member records.

    ``test_only`` makes a nominal mismatch evaluate to ``false`` instead of
    raising ``CastError``. The labels are statically selected source type names
    for a failed ordinary cast.
    """

    location: Location
    nominal: NominalId
    value: "IrExpr"
    test_only: bool
    source_label: str
    target_label: str


@dataclass(frozen=True, slots=True)
class IrNominalIs:
    """IR nominal-member test (``is`` / ``is not``)."""

    location: Location
    nominal: NominalId
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
class IrReturn:
    """IR return: evaluate ``value`` and exit the current function call.

    Implemented by raising an internal ``_ReturnSignal`` Python exception in the
    evaluator; the signal propagates through loops and ``IrTry`` bodies to the
    function-call boundary, where its payload becomes the call result.
    """

    location: Location
    value: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrCatchHandler:
    """A single catch handler in an ``IrTry`` node.

    ``nominal`` identifies the exception type and ``display_name`` is rendering
    metadata:
    - ``nominal=None, display_name=None`` — catch-all (catches everything).
    - ``nominal`` set, ``display_name`` set — specific exact match by
      module-qualified ``ExceptionValue.nominal``.

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
      matches (catch-all when ``nominal is None``; specific when
      ``nominal == exc.nominal``) wins.
    - If a handler matches and ``handler.symbol`` is not ``None``, bind the
      caught ``ExceptionValue`` in the current frame under that symbol.
    - Evaluate the handler's ``body`` and return its value.
    - If no handler matches, re-raise the original ``AglRaise`` unchanged.
    """

    location: Location
    body: "IrExpr"
    handlers: "tuple[IrCatchHandler, ...]"


# ---------------------------------------------------------------------------
# One-level case keys and arms
# ---------------------------------------------------------------------------


class IrLiteralKind(enum.Enum):
    """Runtime equality families supported by scalar case keys."""

    NUMERIC = "numeric"
    BOOL = "bool"
    TEXT = "text"
    NULL = "null"


IrLiteralScalar: TypeAlias = int | decimal.Decimal | bool | str | None


def is_canonical_literal_scalar(kind: IrLiteralKind, value: IrLiteralScalar) -> bool:
    """Report whether *value* is the canonical stored scalar for *kind*.

    Canonical means post-normalization: a ``NUMERIC`` key stores a finite
    :class:`decimal.Decimal` (never an ``int`` or a ``bool``), so this is the
    single source of truth for the shape an :class:`IrLiteralCaseKey` holds.
    """
    return (
        kind is IrLiteralKind.NUMERIC
        and isinstance(value, decimal.Decimal)
        and value.is_finite()
        or kind is IrLiteralKind.BOOL
        and isinstance(value, bool)
        or kind is IrLiteralKind.TEXT
        and isinstance(value, str)
        or kind is IrLiteralKind.NULL
        and value is None
    )


@dataclass(frozen=True, slots=True)
class IrNominalCaseKey:
    """One record-member discriminant identified by nominal declaration."""

    nominal: NominalId


@dataclass(frozen=True, slots=True)
class IrLiteralCaseKey:
    """One canonical scalar discriminant using runtime equality semantics.

    Numeric keys accept an integer or decimal input but store a
    :class:`decimal.Decimal`, so equal integer/decimal spellings are identical
    keys before validation or evaluation.
    """

    kind: IrLiteralKind
    scalar_value: IrLiteralScalar

    def __post_init__(self) -> None:
        value = self.scalar_value
        if (
            self.kind is IrLiteralKind.NUMERIC
            and not isinstance(value, bool)
            and isinstance(value, int)
        ):
            value = decimal.Decimal(value)
            object.__setattr__(self, "scalar_value", value)
        if not is_canonical_literal_scalar(self.kind, value):
            raise ValueError(f"invalid scalar {value!r} for literal case kind {self.kind.name!r}")


IrCaseKey: TypeAlias = IrNominalCaseKey | IrLiteralCaseKey


@dataclass(frozen=True, slots=True)
class IrCaseArm:
    """A single arm in an ``IrCase`` node — NOT a member of ``IrExpr``.

    Enum arms copy their demanded immediate fields into ``field_bindings``.
    Literal arms carry no field bindings. ``body`` may contain another
    one-level ``IrCase``.
    """

    key: IrCaseKey
    field_bindings: tuple[tuple[str, SymbolId], ...]
    body: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrCase:
    """A typeless one-level switch over one already-available value.

    ``default`` represents the compiled decision DAG's remainder edge. A
    well-lowered program never reaches a switch with neither a matching key nor
    a default.
    """

    location: Location
    subject: "IrExpr"
    arms: "tuple[IrCaseArm, ...]"
    default: "IrExpr | None"


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
    million-element array or a ``do[n]`` with a large ``n`` must never be cut
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

    ``kind`` selects array / dict-keys / text iteration.
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
class IrCopyValue:
    """IR host-op for deep or shallow copying, selected by ``kind``.

    Deep copying recursively rebuilds every reachable container with an
    identity memo, preserving sharing and terminating on cycles. Shallow
    copying rebuilds one container level while retaining nested references.
    """

    location: Location
    kind: CopyKind
    value: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrAsk:
    """IR host-op: ask(prompt, agent:, on_parse_error:) builtin call.

    Evaluates ``agent`` (an ``Agent`` enum value), ``prompt`` (text), dispatches
    through the value-driven agent runtime, parses the response via the contract,
    and returns the typed Value.

    ``max_attempts``  — 1 for Abort/absent, 1+n for Retry(n).
    """

    location: Location
    agent: "IrExpr"
    prompt: "IrExpr"
    contract_id: "ContractId"
    max_attempts: int


@dataclass(frozen=True, slots=True)
class IrSessionOpen:
    """IR host-op: open a named or anonymous session for an agent.

    ``transport`` is absent when the source omits its optional transport
    argument. ``name`` always holds an expression, including the empty-text
    default, so the host receives a concrete session name.
    """

    location: Location
    agent: "IrExpr"
    transport: "IrExpr | None"
    name: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrSessionDefault:
    """IR host-op: obtain the lazily managed default session.

    Evaluation reads the current ``default-agent`` register; the host creates
    its default session from that agent once, then returns the same snapshot.
    """

    location: Location


@dataclass(frozen=True, slots=True)
class IrSessionAsk:
    """IR host-op: send a prompt through an existing session.

    Free ``ask`` supplies an :class:`IrSessionDefault` session expression;
    explicit ``Session::ask`` supplies its receiver. The response contract and
    retry count have the same meaning as on :class:`IrAsk`; the session
    supplies the agent selection.
    """

    location: Location
    session: "IrExpr"
    prompt: "IrExpr"
    contract_id: "ContractId"
    max_attempts: int


class IrSessionOpKind(enum.StrEnum):
    """The non-ask lifecycle operations an :class:`IrSessionOp` can carry."""

    COMPACT = "compact"
    RESET = "reset"
    FORK = "fork"
    STATS = "stats"
    SET_NAME = "set-name"
    CLOSE = "close"


@dataclass(frozen=True, slots=True)
class IrSessionOp:
    """IR host-op for a non-ask session operation.

    ``arg`` is optional only for ``compact`` and required for ``set-name``.
    """

    location: Location
    session: "IrExpr"
    op: IrSessionOpKind
    arg: "IrExpr | None" = None


@dataclass(frozen=True, slots=True)
class IrAskRequest:
    """IR host-op: ``ask-request(prompt)`` builtin call.

    Builds the agent-independent AgentRequest record value. The record's
    contract fields are fixed constants describing a text request. Unlike ``IrAsk`` and
    ``IrExec``, this node carries no contract id and no retry count — there is
    nothing to dispatch and no output to parse.
    """

    location: Location
    agent: "IrExpr"
    prompt: "IrExpr"


@dataclass(frozen=True, slots=True)
class IrExec:
    """IR host-op: exec(command, env:, cwd:, timeout:, ...) builtin call."""

    location: Location
    command: "IrExpr"
    env: "IrExpr"
    cwd: "IrExpr"
    timeout: "IrExpr"
    contract_id: "ContractId"
    max_attempts: int


# ---------------------------------------------------------------------------
# Builtin-var register access nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IrBuiltinLoad:
    """IR read of a host-backed ``builtin var`` binding.

    ``key`` identifies the declaration by its owning module, scope path, and name.
    """

    location: Location
    key: BuiltinVarKey | str


@dataclass(frozen=True, slots=True)
class IrBuiltinStore:
    """IR write of a host-backed ``builtin var`` binding.

    ``key`` identifies the declaration by its owning module, scope path, and name.
    Root ``std/config`` engine keys additionally apply their live engine effect.
    """

    location: Location
    key: BuiltinVarKey | str
    value: "IrExpr"


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
    | IrResource
    | IrConstUnit
    | IrConstJsonNull
    | IrMakeArray
    | IrMakeDict
    | IrMakeJsonArray
    | IrMakeJsonObject
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
    | IrUpdateRecord
    | IrIndex
    | IrIndexSet
    | IrRenderTemplate
    | IrMakeRecord
    | IrMakeException
    | IrMakeConstructor
    | IrNominalCast
    | IrNominalIs
    | IrConvert
    | IrIf
    | IrRaise
    | IrReturn
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
    | IrCopyValue
    | IrAsk
    | IrAskRequest
    | IrSessionOpen
    | IrSessionDefault
    | IrSessionAsk
    | IrSessionOp
    | IrExec
    | IrBuiltinLoad
    | IrBuiltinStore
)
