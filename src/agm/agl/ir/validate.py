"""Structural IR validator for the AgL typeless execution IR.

Two tiers (validate_ir runs ONLY when explicitly called):

- **cheap** — node-local structural invariants that require no program tables:
    * Location fields: ``start_offset >= 0``, ``start_line >= 1``,
      ``start_col >= 0``, ``start_offset <= end_offset``.
    * ``IrSequence`` and ``IrBlock`` must be non-empty.

- **deep** — cheap checks PLUS cross-reference checks against the
  ``ExecutableProgram`` tables:
    1. ``program.entry_module`` exists in ``program.modules``; each
       ``ExecutableModule.module_id`` equals its dict key.
    2. Each ``program.symbols`` entry: ``descriptor.symbol_id`` equals its
       key; ``descriptor.owner``, when a ``ModuleId``, exists in
       ``program.modules``; a ``FunctionId`` owner must exist in the
       functions or externs table.
    3. Each ``program.nominals`` entry: ``descriptor.nominal`` equals its key.
    4. Every ``SymbolId`` referenced by ``IrLoad``/``IrBind``/``IrAssign``
       exists in ``program.symbols``.
    5. The root symbol of every ``IrAssign`` is mutable (``mutable=True``).
    6. Every ``Location`` on every node (and ``IrIndexStep``): its
       ``source_id`` exists in ``program.sources``; and
       ``0 <= start_offset <= end_offset <= len(normalized_text)``.
    7. ``program.functions`` and ``program.externs`` share one ``FunctionId``
       space: every id lives in exactly one of the two tables. A reference
       from ``IrMakeClosure``/``IrDirectCall`` (or a symbol owner) is accepted
       against either table; each extern's boundary contract is checked for
       internal consistency (registered nominals, type-variable positions
       matching its declared type parameters).

The expression dispatcher uses a closed structural ``match`` with a final
``assert_never(node)`` arm so that adding an ``IrExpr`` variant in a
future change without a validator arm produces a mypy exhaustiveness error.

``validate_ir`` raises ``InvalidIrError`` on the *first* violation found.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import assert_never

from agm.agl.ir.contracts import (
    BoundaryDict,
    BoundaryEnum,
    BoundaryException,
    BoundaryList,
    BoundaryRecord,
    BoundaryRef,
    BoundaryScalar,
    BoundarySchema,
    BoundarySealVar,
    BoundaryUnit,
    ContractRequest,
    ConversionStrategy,
    DecodeSchema,
    DictDecode,
    EnumDecode,
    ExternContract,
    ListDecode,
    RecordDecode,
    RefDecode,
    ScalarDecode,
)
from agm.agl.ir.ids import ContractId, FunctionId, Location, NominalId, SourceId
from agm.agl.ir.nodes import (
    AutoTraceField,
    IrAgentHandle,
    IrAnd,
    IrArith,
    IrAsk,
    IrAskRequest,
    IrAssign,
    IrBind,
    IrBindPlan,
    IrBlock,
    IrBreak,
    IrCase,
    IrCatchHandler,
    IrCoerce,
    IrCompare,
    IrConfigBind,
    IrConstBool,
    IrConstDecimal,
    IrConstInt,
    IrConstJsonNull,
    IrConstructorPlan,
    IrConstText,
    IrConstUnit,
    IrContains,
    IrContinue,
    IrConvert,
    IrDirectCall,
    IrExec,
    IrExpr,
    IrField,
    IrFunctionParam,
    IrIf,
    IrIndex,
    IrIndexStep,
    IrIndirectCall,
    IrIterHasNext,
    IrIterInit,
    IrIterNext,
    IrLiteralPlan,
    IrLoad,
    IrLoop,
    IrMakeClosure,
    IrMakeConstructor,
    IrMakeDict,
    IrMakeEnum,
    IrMakeException,
    IrMakeList,
    IrMakeRecord,
    IrMatchPlan,
    IrOr,
    IrParseJson,
    IrPrint,
    IrRaise,
    IrRenderTemplate,
    IrRenderValue,
    IrReturn,
    IrSequence,
    IrTemplateText,
    IrTemplateValue,
    IrTry,
    IrUnary,
    IrVariantIs,
    IrVariantPlan,
    IrWildcardPlan,
    UseDefault,
)
from agm.agl.ir.operations import ArithKind, ArithOp, CmpOp, CompareKind, UnaryOp
from agm.agl.ir.program import ExecutableProgram, IrParam, NominalKind, SourceFile
from agm.agl.modules.ids import ModuleId

__all__ = ["InvalidIrError", "validate_ir"]


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class InvalidIrError(Exception):
    """Raised by ``validate_ir`` when a structural invariant is violated.

    The message identifies the offending node or table entry so the caller
    can diagnose the problem without inspecting the full program.
    """


# ---------------------------------------------------------------------------
# Internal context — passed through recursive calls
# ---------------------------------------------------------------------------


class _Context:
    """Collects all program-level tables needed by deep checks.

    Kept as a small ``__slots__`` helper object rather than globals so that the
    validator is re-entrant and thread-safe.
    """

    __slots__ = ("program", "deep")

    def __init__(self, program: ExecutableProgram, *, deep: bool) -> None:
        self.program = program
        self.deep = deep


# ---------------------------------------------------------------------------
# Location validation helpers
# ---------------------------------------------------------------------------


def _check_location_cheap(loc: Location) -> None:
    """Validate local structural invariants on a ``Location``."""
    if loc.start_offset < 0:
        raise InvalidIrError(
            f"Location has negative start_offset={loc.start_offset!r}"
        )
    if loc.start_offset > loc.end_offset:
        raise InvalidIrError(
            f"Location has start_offset={loc.start_offset!r} > end_offset={loc.end_offset!r}"
        )
    if loc.start_line < 1:
        raise InvalidIrError(
            f"Location has start_line={loc.start_line!r} (must be >= 1)"
        )
    if loc.start_col < 0:
        raise InvalidIrError(
            f"Location has negative start_col={loc.start_col!r}"
        )


def _check_location_deep(loc: Location, ctx: _Context) -> None:
    """Validate cross-reference invariants on a ``Location`` (deep tier)."""
    source_id: SourceId = loc.source_id
    if source_id not in ctx.program.sources:
        raise InvalidIrError(
            f"Location references source_id={source_id!r} which is not in program.sources"
        )
    source: SourceFile = ctx.program.sources[source_id]
    text_len = len(source.normalized_text)
    if loc.end_offset > text_len:
        raise InvalidIrError(
            f"Location has end_offset={loc.end_offset!r} which exceeds"
            f" source length {text_len!r} for source_id={source_id!r}"
        )


def _validate_location(loc: Location, ctx: _Context) -> None:
    """Run cheap (and optionally deep) location checks."""
    _check_location_cheap(loc)
    if ctx.deep:
        _check_location_deep(loc, ctx)


# ---------------------------------------------------------------------------
# Nominal completeness check helpers (deep tier)
# ---------------------------------------------------------------------------


def _check_nominal_in_table(nominal: NominalId, ctx: _Context) -> None:
    """Raise ``InvalidIrError`` if *nominal* is not in ``program.nominals``."""
    if nominal not in ctx.program.nominals:
        raise InvalidIrError(
            f"IR node references nominal {nominal!r} which is not in program.nominals"
        )


def _check_enum_variant(nominal: NominalId, variant: str, ctx: _Context) -> None:
    """Raise ``InvalidIrError`` if *variant* is not declared in the nominal's descriptor."""
    desc = ctx.program.nominals.get(nominal)
    if desc is None:  # pragma: no cover
        return  # already caught by _check_nominal_in_table
    if desc.kind is not NominalKind.ENUM:
        return  # not an enum; variant check not applicable
    known = {v.name for v in desc.variants}
    if variant not in known:
        raise InvalidIrError(
            f"IR node references variant {variant!r} of {nominal!r}"
            f" which is not in the descriptor's variants {sorted(known)!r}"
        )


_DECODE_STRATEGIES = frozenset(
    {
        ConversionStrategy.NARROW_DECIMAL_TO_INT,
        ConversionStrategy.PARSE_TEXT_THEN_DECODE,
        ConversionStrategy.DECODE_JSON,
    }
)


def _check_recipe_consistency(
    strategy: ConversionStrategy,
    json_schema: str | None,
    decode: DecodeSchema | None,
    defs: "tuple[tuple[str, DecodeSchema], ...]",
) -> None:
    """Enforce that decode strategies carry schema+decode and total strategies do not."""
    needs_decode = strategy in _DECODE_STRATEGIES
    has_decode = json_schema is not None and decode is not None
    if needs_decode and not has_decode:
        raise InvalidIrError(
            f"ConversionRecipe strategy {strategy.value!r} requires json_schema and decode"
        )
    if not needs_decode and (json_schema is not None or decode is not None or defs):
        raise InvalidIrError(
            f"ConversionRecipe strategy {strategy.value!r} must not carry "
            "json_schema/decode/defs"
        )


def _check_decode_nominals(
    decode: DecodeSchema, defs: "tuple[tuple[str, DecodeSchema], ...]", ctx: _Context
) -> None:
    """Deep tier: every nominal referenced by a decode schema (root + ``defs``) must be registered.

    Walks *decode* (the root) and every DISTINCT ``defs`` entry exactly once:
    a ``RefDecode`` node is checked for key membership and its ref chain is
    required to reach a non-ref body, but that body is not walked inline from
    the ref — each entry is instead walked once from the loop below, so a
    normal self- or mutually-recursive decode body terminates while malformed
    ref-only cycles are rejected.
    """
    visited: set[str] = set()
    for key, _entry in defs:
        if key in visited:
            raise InvalidIrError(f"DecodeSchema has duplicate $defs key {key!r}")
        visited.add(key)
    defs_map = dict(defs)
    _walk_decode_schema(decode, defs_map, ctx)
    for key, entry in defs:
        _walk_decode_schema(entry, defs_map, ctx)


def _walk_decode_schema(
    decode: DecodeSchema, defs: "Mapping[str, DecodeSchema]", ctx: _Context
) -> None:
    """Walk one decode-schema node (never re-entering a ``RefDecode`` target)."""
    match decode:
        case ScalarDecode():
            return
        case RefDecode(key=key):
            _check_refdecode_chain(key, defs)
        case ListDecode(elem=elem):
            _walk_decode_schema(elem, defs, ctx)
        case DictDecode(value=value_schema):
            _walk_decode_schema(value_schema, defs, ctx)
        case RecordDecode(nominal=nominal, fields=fields):
            _check_nominal_in_table(nominal, ctx)
            for _fname, fschema in fields:
                _walk_decode_schema(fschema, defs, ctx)
        case EnumDecode(nominal=nominal, variants=variants):
            _check_nominal_in_table(nominal, ctx)
            for variant in variants:
                for _fname, fschema in variant.fields:
                    _walk_decode_schema(fschema, defs, ctx)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _check_refdecode_chain(key: str, defs: Mapping[str, DecodeSchema]) -> None:
    """Ensure a ``RefDecode`` chain reaches a non-ref body without cycling."""
    seen: set[str] = set()
    current = key
    while True:
        if current in seen:
            raise InvalidIrError(
                f"DecodeSchema RefDecode cycle reaches no body at $defs key {current!r}"
            )
        seen.add(current)
        target = defs.get(current)
        if target is None:
            raise InvalidIrError(
                f"DecodeSchema RefDecode references unknown $defs key {current!r}"
            )
        if not isinstance(target, RefDecode):
            return
        current = target.key


# ---------------------------------------------------------------------------
# Extern boundary contract checks (deep tier)
# ---------------------------------------------------------------------------


def _check_boundary_schema_nominals(
    schema: BoundarySchema, defs: "Mapping[str, BoundarySchema]", ctx: _Context
) -> None:
    """Deep tier: every nominal referenced by a boundary schema must be registered.

    Never re-enters a ``BoundaryRef`` target — each ``defs`` body is walked once
    by :func:`_validate_extern_contract`; a ref only checks its chain resolves.
    """
    match schema:
        case BoundaryScalar() | BoundaryUnit() | BoundarySealVar():
            return
        case BoundaryRef(key=key):
            _check_boundaryref_chain(key, defs)
        case BoundaryList(element=element):
            _check_boundary_schema_nominals(element, defs, ctx)
        case BoundaryDict(value=value_schema):
            _check_boundary_schema_nominals(value_schema, defs, ctx)
        case BoundaryRecord(nominal=nominal, fields=fields):
            _check_nominal_in_table(nominal, ctx)
            for _fname, fschema in fields:
                _check_boundary_schema_nominals(fschema, defs, ctx)
        case BoundaryEnum(nominal=nominal, variants=variants):
            _check_nominal_in_table(nominal, ctx)
            for variant in variants:
                for _fname, fschema in variant.fields:
                    _check_boundary_schema_nominals(fschema, defs, ctx)
        case BoundaryException(nominal=nominal, fields=fields):
            _check_nominal_in_table(nominal, ctx)
            for _fname, fschema in fields:
                _check_boundary_schema_nominals(fschema, defs, ctx)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _check_boundaryref_chain(key: str, defs: "Mapping[str, BoundarySchema]") -> None:
    """Ensure a ``BoundaryRef`` chain reaches a non-ref body without cycling."""
    seen: set[str] = set()
    current = key
    while True:
        if current in seen:
            raise InvalidIrError(
                f"BoundarySchema BoundaryRef cycle reaches no body at defs key {current!r}"
            )
        seen.add(current)
        target = defs.get(current)
        if target is None:
            raise InvalidIrError(
                f"BoundarySchema BoundaryRef references unknown defs key {current!r}"
            )
        if not isinstance(target, BoundaryRef):
            return
        current = target.key


def _collect_boundary_seal_vars(schema: BoundarySchema, out: set[str]) -> None:
    """Collect every ``BoundarySealVar.var`` name appearing directly in *schema*.

    Does not follow a ``BoundaryRef`` — every ``defs`` body is scanned for seal
    vars separately by :func:`_validate_extern_contract`.
    """
    match schema:
        case BoundaryScalar() | BoundaryUnit() | BoundaryRef():
            return
        case BoundarySealVar(var=var):
            out.add(var)
        case BoundaryList(element=element):
            _collect_boundary_seal_vars(element, out)
        case BoundaryDict(value=value_schema):
            _collect_boundary_seal_vars(value_schema, out)
        case BoundaryRecord(fields=fields):
            for _fname, fschema in fields:
                _collect_boundary_seal_vars(fschema, out)
        case BoundaryEnum(variants=variants):
            for variant in variants:
                for _fname, fschema in variant.fields:
                    _collect_boundary_seal_vars(fschema, out)
        case BoundaryException(fields=fields):
            for _fname, fschema in fields:
                _collect_boundary_seal_vars(fschema, out)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _validate_extern_contract(
    fn_key: FunctionId, contract: ExternContract, ctx: _Context
) -> None:
    """Validate one extern's boundary contract (deep tier).

    Every nominal referenced by a param/result schema (or a shared ``defs``
    body) must be registered, every ``BoundaryRef`` must resolve to a body in
    ``defs`` without cycling, and every type-variable position
    (``BoundarySealVar``) must name one of the contract's declared
    ``type_params``.
    """
    seen_keys: set[str] = set()
    for key, _entry in contract.defs:
        if key in seen_keys:
            raise InvalidIrError(
                f"ExternFunctionDescriptor for {fn_key!r}: contract has duplicate"
                f" defs key {key!r}"
            )
        seen_keys.add(key)
    defs_map = dict(contract.defs)

    seal_vars: set[str] = set()
    for param_schema in contract.params:
        _check_boundary_schema_nominals(param_schema.schema, defs_map, ctx)
        _collect_boundary_seal_vars(param_schema.schema, seal_vars)
    _check_boundary_schema_nominals(contract.result, defs_map, ctx)
    _collect_boundary_seal_vars(contract.result, seal_vars)
    for _key, entry in contract.defs:
        _check_boundary_schema_nominals(entry, defs_map, ctx)
        _collect_boundary_seal_vars(entry, seal_vars)
    unknown = seal_vars - set(contract.type_params)
    if unknown:
        raise InvalidIrError(
            f"ExternFunctionDescriptor for {fn_key!r}: contract references type"
            f" variable(s) {sorted(unknown)!r} not declared in type_params"
            f" {contract.type_params!r}"
        )


# ---------------------------------------------------------------------------
# Function-id membership (shared between program.functions and program.externs)
# ---------------------------------------------------------------------------


def _fn_or_extern_exists(fn_id: FunctionId, program: ExecutableProgram) -> bool:
    """Return ``True`` if *fn_id* is registered in the functions or externs table."""
    return fn_id in program.functions or fn_id in program.externs


def _resolve_callable_params(
    fn_id: FunctionId, ctx: _Context, node_desc: str
) -> "tuple[IrFunctionParam, ...]":
    """Resolve *fn_id* to its declared parameter tuple.

    ``function_id``s share one id space between ``program.functions`` and
    ``program.externs`` — a reference resolving to neither table is a
    dangling reference; the tables are checked disjoint separately (an id
    registered in both is a global table-consistency error, not a
    per-reference one).
    """
    fn_desc = ctx.program.functions.get(fn_id)
    if fn_desc is not None:
        return fn_desc.params
    extern_desc = ctx.program.externs.get(fn_id)
    if extern_desc is not None:
        return extern_desc.params
    raise InvalidIrError(
        f"{node_desc} references function_id={fn_id!r} which is not in"
        " program.functions or program.externs"
    )


# ---------------------------------------------------------------------------
# IrIndexStep validation
# ---------------------------------------------------------------------------


def _validate_index_step(step: IrIndexStep, ctx: _Context) -> None:
    _validate_location(step.location, ctx)
    _validate_expr(step.index, ctx)


# ---------------------------------------------------------------------------
# IrCatchHandler validation (deep tier)
# ---------------------------------------------------------------------------


def _validate_catch_handler(handler: IrCatchHandler, ctx: _Context) -> None:
    """Validate a catch handler: nominal/symbol cross-references (deep) + body."""
    if ctx.deep:
        if handler.nominal is not None:
            _check_nominal_in_table(handler.nominal, ctx)
        if handler.symbol is not None:
            if handler.symbol not in ctx.program.symbols:
                raise InvalidIrError(
                    f"IrCatchHandler references symbol_id={handler.symbol.value!r}"
                    " which is not in program.symbols"
                )
    _validate_expr(handler.body, ctx)


# ---------------------------------------------------------------------------
# IrMatchPlan validator (closed union)
# ---------------------------------------------------------------------------


def _validate_match_plan(plan: IrMatchPlan, ctx: _Context) -> None:
    """Validate a closed ``IrMatchPlan`` node recursively.

    The final ``assert_never`` arm ensures mypy reports a type error when
    a new ``IrMatchPlan`` variant is added without a corresponding arm here.
    """
    match plan:
        case IrWildcardPlan():
            pass

        case IrBindPlan(symbol=sym):
            if ctx.deep:
                if sym not in ctx.program.symbols:
                    raise InvalidIrError(
                        f"IrBindPlan references symbol_id={sym.value!r}"
                        " which is not in program.symbols"
                    )

        case IrLiteralPlan(value=val_expr):
            _validate_expr(val_expr, ctx)

        case IrVariantPlan():
            pass

        case IrConstructorPlan(fields=fields):
            for _fname, subplan in fields:
                _validate_match_plan(subplan, ctx)

        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


# ---------------------------------------------------------------------------
# Closed-union expression dispatcher
# ---------------------------------------------------------------------------


def _validate_expr(node: IrExpr, ctx: _Context) -> None:
    """Dispatch validation over the closed ``IrExpr`` union.

    The final ``assert_never`` arm ensures mypy reports a type error when a
    new ``IrExpr`` variant is added without a corresponding arm here.
    """
    match node:
        case IrConstInt():
            _validate_location(node.location, ctx)

        case IrConstDecimal():
            _validate_location(node.location, ctx)

        case IrConstBool():
            _validate_location(node.location, ctx)

        case IrConstText():
            _validate_location(node.location, ctx)

        case IrConstUnit():
            _validate_location(node.location, ctx)

        case IrConstJsonNull():
            _validate_location(node.location, ctx)

        case IrMakeList():
            _validate_location(node.location, ctx)
            for item in node.items:
                _validate_expr(item, ctx)

        case IrMakeDict():
            _validate_location(node.location, ctx)
            for key_expr, val_expr in node.entries:
                _validate_expr(key_expr, ctx)
                _validate_expr(val_expr, ctx)

        case IrLoad():
            _validate_location(node.location, ctx)
            if ctx.deep:
                if node.symbol not in ctx.program.symbols:
                    raise InvalidIrError(
                        f"IrLoad references symbol_id={node.symbol.value!r}"
                        " which is not in program.symbols"
                    )

        case IrBind():
            _validate_location(node.location, ctx)
            if ctx.deep:
                if node.symbol not in ctx.program.symbols:
                    raise InvalidIrError(
                        f"IrBind references symbol_id={node.symbol.value!r}"
                        " which is not in program.symbols"
                    )
            _validate_expr(node.value, ctx)

        case IrAssign():
            _validate_location(node.location, ctx)
            if ctx.deep:
                if node.symbol not in ctx.program.symbols:
                    raise InvalidIrError(
                        f"IrAssign references symbol_id={node.symbol.value!r}"
                        " which is not in program.symbols"
                    )
                desc = ctx.program.symbols[node.symbol]
                if not desc.mutable:
                    raise InvalidIrError(
                        f"IrAssign targets symbol_id={node.symbol.value!r}"
                        f" (public_name={desc.public_name!r}) which is not mutable"
                    )
            for step in node.path:
                _validate_index_step(step, ctx)
            _validate_expr(node.value, ctx)

        case IrCoerce():
            _validate_location(node.location, ctx)
            _validate_expr(node.value, ctx)

        case IrSequence():
            _validate_location(node.location, ctx)
            if len(node.items) == 0:
                raise InvalidIrError("IrSequence must be non-empty (items is empty)")
            for item in node.items:
                _validate_expr(item, ctx)

        case IrBlock():
            _validate_location(node.location, ctx)
            if len(node.items) == 0:
                raise InvalidIrError("IrBlock must be non-empty (items is empty)")
            for item in node.items:
                _validate_expr(item, ctx)

        case IrArith(op=op, kind=kind, lhs=lhs, rhs=rhs):
            _validate_location(node.location, ctx)
            # TEXT kind is only valid with ADD
            if kind is ArithKind.TEXT and op is not ArithOp.ADD:
                raise InvalidIrError(
                    f"IrArith: TEXT kind is only valid with ADD, got op={op!r}"
                )
            # DIV op requires DECIMAL kind (DIV always returns decimal)
            if op is ArithOp.DIV and kind is not ArithKind.DECIMAL:
                raise InvalidIrError(
                    f"IrArith: DIV op requires DECIMAL kind, got kind={kind!r}"
                )
            _validate_expr(lhs, ctx)
            _validate_expr(rhs, ctx)

        case IrCompare(op=op, kind=kind, lhs=lhs, rhs=rhs):
            _validate_location(node.location, ctx)
            # EQ/NEQ requires STRUCTURAL kind
            if op in (CmpOp.EQ, CmpOp.NEQ) and kind is not CompareKind.STRUCTURAL:
                raise InvalidIrError(
                    f"IrCompare: EQ/NEQ requires STRUCTURAL kind, got kind={kind!r}"
                )
            # Ordering ops (LT/LE/GT/GE) require a non-STRUCTURAL kind
            if op in (CmpOp.LT, CmpOp.LE, CmpOp.GT, CmpOp.GE) and kind is CompareKind.STRUCTURAL:
                raise InvalidIrError(
                    f"IrCompare: ordering op {op!r} requires INT/DECIMAL/TEXT kind,"
                    f" got STRUCTURAL"
                )
            _validate_expr(lhs, ctx)
            _validate_expr(rhs, ctx)

        case IrContains(kind=_kind, item=item, container=container):
            _validate_location(node.location, ctx)
            _validate_expr(item, ctx)
            _validate_expr(container, ctx)

        case IrAnd(lhs=lhs, rhs=rhs):
            _validate_location(node.location, ctx)
            _validate_expr(lhs, ctx)
            _validate_expr(rhs, ctx)

        case IrOr(lhs=lhs, rhs=rhs):
            _validate_location(node.location, ctx)
            _validate_expr(lhs, ctx)
            _validate_expr(rhs, ctx)

        case IrUnary(op=op, kind=kind, value=val):
            _validate_location(node.location, ctx)
            # NOT requires kind=None; NEG requires kind set
            if op is UnaryOp.NOT and kind is not None:
                raise InvalidIrError(
                    f"IrUnary NOT: kind must be None, got kind={kind!r}"
                )
            if op is UnaryOp.NEG and kind is None:
                raise InvalidIrError(
                    "IrUnary NEG: kind must not be None"
                )
            _validate_expr(val, ctx)

        case IrField(value=val):
            _validate_location(node.location, ctx)
            _validate_expr(val, ctx)

        case IrIndex(kind=_kind, value=val, index=idx):
            _validate_location(node.location, ctx)
            _validate_expr(val, ctx)
            _validate_expr(idx, ctx)

        case IrRenderTemplate(segments=segs):
            _validate_location(node.location, ctx)
            for seg in segs:
                match seg:
                    case IrTemplateText():
                        pass
                    case IrTemplateValue(value=val):
                        _validate_expr(val, ctx)
                    case _ as unreachable_seg:  # pragma: no cover
                        assert_never(unreachable_seg)

        case IrMakeRecord(nominal=nominal, fields=fields):
            _validate_location(node.location, ctx)
            if ctx.deep:
                _check_nominal_in_table(nominal, ctx)
            for _fname, fexpr in fields:
                _validate_expr(fexpr, ctx)

        case IrMakeEnum(nominal=nominal, variant=variant, fields=fields):
            _validate_location(node.location, ctx)
            if ctx.deep:
                _check_nominal_in_table(nominal, ctx)
                _check_enum_variant(nominal, variant, ctx)
            for _fname, fexpr in fields:
                _validate_expr(fexpr, ctx)

        case IrMakeException(nominal=nominal, fields=fields):
            _validate_location(node.location, ctx)
            if ctx.deep:
                _check_nominal_in_table(nominal, ctx)
            for _fname, slot in fields:
                if isinstance(slot, AutoTraceField):
                    pass
                else:
                    _validate_expr(slot, ctx)

        case IrMakeConstructor(nominal=nominal, variant=variant):
            _validate_location(node.location, ctx)
            if ctx.deep:
                _check_nominal_in_table(nominal, ctx)
                if variant is not None:
                    _check_enum_variant(nominal, variant, ctx)

        case IrVariantIs(nominal=nominal, variant=variant, value=val):
            _validate_location(node.location, ctx)
            if ctx.deep:
                _check_nominal_in_table(nominal, ctx)
                _check_enum_variant(nominal, variant, ctx)
            _validate_expr(val, ctx)

        case IrConvert(value=val, recipe=recipe):
            _validate_location(node.location, ctx)
            _check_recipe_consistency(
                recipe.strategy, recipe.json_schema, recipe.decode, recipe.defs
            )
            if ctx.deep and recipe.decode is not None:
                _check_decode_nominals(recipe.decode, recipe.defs, ctx)
            _validate_expr(val, ctx)

        case IrIf(branches=branches):
            _validate_location(node.location, ctx)
            for branch in branches:
                if branch.cond is not None:
                    _validate_expr(branch.cond, ctx)
                _validate_expr(branch.body, ctx)

        case IrRaise(exc=exc):
            _validate_location(node.location, ctx)
            _validate_expr(exc, ctx)

        case IrReturn(value=value):
            _validate_location(node.location, ctx)
            _validate_expr(value, ctx)

        case IrTry(body=body, handlers=handlers):
            _validate_location(node.location, ctx)
            _validate_expr(body, ctx)
            for handler in handlers:
                _validate_catch_handler(handler, ctx)

        case IrCase(subject=subject, arms=arms):
            _validate_location(node.location, ctx)
            _validate_expr(subject, ctx)
            for arm in arms:
                _validate_match_plan(arm.plan, ctx)
                _validate_expr(arm.body, ctx)

        case IrLoop(body=body):
            _validate_location(node.location, ctx)
            _validate_expr(body, ctx)

        case IrBreak():
            _validate_location(node.location, ctx)

        case IrContinue():
            _validate_location(node.location, ctx)

        case IrIterInit(collection=collection):
            _validate_location(node.location, ctx)
            _validate_expr(collection, ctx)

        case IrIterHasNext(iterator=iterator):
            _validate_location(node.location, ctx)
            _validate_expr(iterator, ctx)

        case IrIterNext(iterator=iterator):
            _validate_location(node.location, ctx)
            _validate_expr(iterator, ctx)

        case IrMakeClosure(function_id=fn_id, captures=captures):
            _validate_location(node.location, ctx)
            if ctx.deep:
                if not _fn_or_extern_exists(fn_id, ctx.program):
                    raise InvalidIrError(
                        f"IrMakeClosure references function_id={fn_id!r}"
                        " which is not in program.functions or program.externs"
                    )
                for cap in captures:
                    if cap.symbol not in ctx.program.symbols:
                        raise InvalidIrError(
                            f"IrMakeClosure capture references symbol_id={cap.symbol!r}"
                            " which is not in program.symbols"
                        )

        case IrDirectCall(function_id=fn_id, arguments=arguments):
            _validate_location(node.location, ctx)
            if ctx.deep:
                params = _resolve_callable_params(fn_id, ctx, "IrDirectCall")
                if len(arguments) != len(params):
                    raise InvalidIrError(
                        f"IrDirectCall to function_id={fn_id!r} has {len(arguments)}"
                        f" arguments but the function has {len(params)} parameters"
                    )
                for index, arg in enumerate(arguments):
                    if isinstance(arg, UseDefault):
                        if arg.param_index != index:
                            raise InvalidIrError(
                                f"IrDirectCall to function_id={fn_id!r}: UseDefault at"
                                f" position {index} has param_index={arg.param_index}"
                                " (must equal its position)"
                            )
                        if params[index].default is None:
                            raise InvalidIrError(
                                f"IrDirectCall to function_id={fn_id!r}: UseDefault for"
                                f" parameter {index} which has no default"
                            )
            for arg in arguments:
                if not isinstance(arg, UseDefault):
                    _validate_expr(arg, ctx)

        case IrIndirectCall(callee=callee, arguments=arguments):
            _validate_location(node.location, ctx)
            _validate_expr(callee, ctx)
            for arg in arguments:
                _validate_expr(arg, ctx)

        case IrPrint(value=val):
            _validate_location(node.location, ctx)
            _validate_expr(val, ctx)

        case IrRenderValue(value=val, pretty=pretty, quote_strings=quote_strings):
            _validate_location(node.location, ctx)
            _validate_expr(val, ctx)
            if pretty is not None:
                _validate_expr(pretty, ctx)
            if quote_strings is not None:
                _validate_expr(quote_strings, ctx)

        case IrParseJson(value=val):
            _validate_location(node.location, ctx)
            _validate_expr(val, ctx)

        case IrAgentHandle():
            _validate_location(node.location, ctx)

        case IrAsk(agent=agent_expr, prompt=prompt_expr, contract_id=contract_id):
            _validate_location(node.location, ctx)
            _validate_expr(agent_expr, ctx)
            _validate_expr(prompt_expr, ctx)
            if ctx.deep:
                if contract_id not in ctx.program.contracts:
                    raise InvalidIrError(
                        f"IrAsk references contract_id={contract_id!r}"
                        " which is not in program.contracts"
                    )
                if node.max_attempts < 1:
                    raise InvalidIrError(
                        f"IrAsk has max_attempts={node.max_attempts!r} (must be >= 1)"
                    )

        case IrAskRequest(agent=agent_expr, prompt=prompt_expr, contract_id=contract_id):
            _validate_location(node.location, ctx)
            _validate_expr(agent_expr, ctx)
            _validate_expr(prompt_expr, ctx)
            if ctx.deep:
                if contract_id not in ctx.program.contracts:
                    raise InvalidIrError(
                        f"IrAskRequest references contract_id={contract_id!r}"
                        " which is not in program.contracts"
                    )
                if node.max_attempts < 1:
                    raise InvalidIrError(
                        f"IrAskRequest has max_attempts={node.max_attempts!r} (must be >= 1)"
                    )

        case IrExec(command=command_expr, contract_id=contract_id):
            _validate_location(node.location, ctx)
            _validate_expr(command_expr, ctx)
            if ctx.deep:
                if contract_id not in ctx.program.contracts:
                    raise InvalidIrError(
                        f"IrExec references contract_id={contract_id!r}"
                        " which is not in program.contracts"
                    )
                if node.max_attempts < 1:
                    raise InvalidIrError(
                        f"IrExec has max_attempts={node.max_attempts!r} (must be >= 1)"
                    )

        case IrConfigBind(symbol=sym, value=value_expr):
            _validate_location(node.location, ctx)
            if ctx.deep:
                if sym not in ctx.program.symbols:
                    raise InvalidIrError(
                        f"IrConfigBind references symbol_id={sym.value!r}"
                        " which is not in program.symbols"
                    )
            if value_expr is not None:
                _validate_expr(value_expr, ctx)

        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


# ---------------------------------------------------------------------------
# Deep tier — program-table checks
# ---------------------------------------------------------------------------


def _validate_program_tables(program: ExecutableProgram) -> None:
    """Run deep cross-reference checks on the top-level program tables."""

    # 1. entry_module
    if program.entry_module not in program.modules:
        raise InvalidIrError(
            f"entry_module={program.entry_module!r} is not in program.modules"
        )

    # 1b. module key/id consistency
    for key, em in program.modules.items():
        if em.module_id != key:
            raise InvalidIrError(
                f"program.modules entry keyed by {key!r} has"
                f" module_id={em.module_id!r} (mismatch)"
            )

    # 2. symbol descriptor consistency
    for sym_key, sym_desc in program.symbols.items():
        if sym_desc.symbol_id != sym_key:
            raise InvalidIrError(
                f"program.symbols entry keyed by {sym_key!r} has"
                f" symbol_id={sym_desc.symbol_id!r} (mismatch)"
            )
        owner = sym_desc.owner
        if isinstance(owner, ModuleId):
            if owner not in program.modules:
                raise InvalidIrError(
                    f"SymbolDescriptor for symbol_id={sym_key!r} has owner={owner!r}"
                    " which is not in program.modules"
                )
        elif isinstance(owner, FunctionId):
            if not _fn_or_extern_exists(owner, program):
                raise InvalidIrError(
                    f"SymbolDescriptor for symbol_id={sym_key!r} has owner={owner!r}"
                    " which is not in program.functions or program.externs"
                )
        else:
            assert_never(owner)  # pragma: no cover

    # 3. nominal descriptor consistency
    for nom_key, nom_desc in program.nominals.items():
        if nom_desc.nominal != nom_key:
            raise InvalidIrError(
                f"program.nominals entry keyed by {nom_key!r} has"
                f" nominal={nom_desc.nominal!r} (mismatch)"
            )

    # 4. functions table consistency
    fn_ctx = _Context(program, deep=True)
    for fn_key, fn_desc in program.functions.items():
        if fn_desc.function_id != fn_key:
            raise InvalidIrError(
                f"program.functions entry keyed by {fn_key!r} has"
                f" function_id={fn_desc.function_id!r} (mismatch)"
            )
        if fn_desc.function_symbol not in program.symbols:
            raise InvalidIrError(
                f"FunctionDescriptor for {fn_key!r} has function_symbol={fn_desc.function_symbol!r}"
                " which is not in program.symbols"
            )
        if fn_desc.module_id not in program.modules:
            raise InvalidIrError(
                f"FunctionDescriptor for {fn_key!r} has module_id={fn_desc.module_id!r}"
                " which is not in program.modules"
            )
        for param in fn_desc.params:
            if param.symbol not in program.symbols:
                raise InvalidIrError(
                    f"FunctionDescriptor for {fn_key!r}: param symbol {param.symbol!r}"
                    " is not in program.symbols"
                )
            if param.default is not None:
                _validate_expr(param.default, fn_ctx)
        _validate_expr(fn_desc.body, fn_ctx)

    # 4b. functions/externs share one id space: an id in both tables is a
    #     compiler bug (which table dispatch resolves through is ambiguous).
    dup_id_set: set[FunctionId] = set(program.functions) & set(program.externs)
    dup_ids = sorted(fid.value for fid in dup_id_set)
    if dup_ids:
        raise InvalidIrError(
            f"function_id(s) {dup_ids!r} registered in both"
            " program.functions and program.externs"
        )

    # 4c. externs table consistency — mirrors the functions table checks above,
    #     plus the boundary contract's internal consistency (nominal references,
    #     type-variable positions matching declared type_params).
    for fn_key, extern_desc in program.externs.items():
        if extern_desc.function_id != fn_key:
            raise InvalidIrError(
                f"program.externs entry keyed by {fn_key!r} has"
                f" function_id={extern_desc.function_id!r} (mismatch)"
            )
        if extern_desc.function_symbol not in program.symbols:
            raise InvalidIrError(
                f"ExternFunctionDescriptor for {fn_key!r} has"
                f" function_symbol={extern_desc.function_symbol!r} which is not"
                " in program.symbols"
            )
        if extern_desc.module_id not in program.modules:
            raise InvalidIrError(
                f"ExternFunctionDescriptor for {fn_key!r} has"
                f" module_id={extern_desc.module_id!r} which is not in program.modules"
            )
        for param in extern_desc.params:
            if param.symbol not in program.symbols:
                raise InvalidIrError(
                    f"ExternFunctionDescriptor for {fn_key!r}: param symbol"
                    f" {param.symbol!r} is not in program.symbols"
                )
            if param.default is not None:
                _validate_expr(param.default, fn_ctx)
        if len(extern_desc.params) != len(extern_desc.contract.params):
            raise InvalidIrError(
                f"ExternFunctionDescriptor for {fn_key!r} has {len(extern_desc.params)}"
                f" IR params but its contract has {len(extern_desc.contract.params)}"
                " boundary params"
            )
        _validate_extern_contract(fn_key, extern_desc.contract, fn_ctx)

    # 5. params table — each IrParam must reference a registered symbol, and
    #    the default expression (if present) must be structurally valid.
    for ir_param in program.params:
        _validate_ir_param(ir_param, fn_ctx)

    # 6. contracts table — each ContractRequest must be consistent.
    for cid, contract_req in program.contracts.items():
        _validate_contract_request(cid, contract_req, fn_ctx)

    # (Sources table has no key/id consistency invariant beyond being keyed by
    # SourceId; key consistency is structural to dict construction.)


def _validate_ir_param(param: IrParam, ctx: _Context) -> None:
    """Validate a single ``IrParam`` descriptor (deep tier)."""
    _validate_location(param.location, ctx)
    if ctx.deep:
        if param.symbol not in ctx.program.symbols:
            raise InvalidIrError(
                f"IrParam public_name={param.public_name!r} references"
                f" symbol_id={param.symbol.value!r} which is not in program.symbols"
            )
        if param.external_decoder is not None:
            _check_decode_nominals(
                param.external_decoder.decode,
                param.external_decoder.defs,
                ctx,
            )
    if param.default is not None:
        _validate_expr(param.default, ctx)


def _validate_contract_request(
    cid: ContractId,
    req: ContractRequest,
    ctx: _Context,
) -> None:
    """Validate a ContractRequest entry (deep tier)."""
    has_decode_fields = req.json_schema is not None or req.decode is not None or bool(req.defs)
    if req.is_unit or req.codec_name == "text":
        if has_decode_fields:
            raise InvalidIrError(
                f"ContractRequest {cid!r} must not carry json_schema/decode/defs"
            )
        return
    if req.codec_name == "json":
        if req.json_schema is None:
            raise InvalidIrError(
                f"ContractRequest {cid!r} has codec_name='json' but json_schema is None"
            )
        if req.decode is None:
            raise InvalidIrError(
                f"ContractRequest {cid!r} has codec_name='json' but decode is None"
            )
    elif req.defs and req.decode is None:
        raise InvalidIrError(f"ContractRequest {cid!r} has defs but decode is None")
    if req.decode is not None:
        _check_decode_nominals(req.decode, req.defs, ctx)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_ir(program: ExecutableProgram, *, deep: bool = True) -> None:
    """Validate the structural integrity of ``program``.

    :param program: the ``ExecutableProgram`` to validate.
    :param deep: when ``True`` (default) runs cheap + deep checks; when
        ``False`` runs only the cheap node-local tier (no table lookups).
    :raises InvalidIrError: on the first violation found, with a message
        identifying the offending node or table entry.
    """
    ctx = _Context(program, deep=deep)

    if deep:
        _validate_program_tables(program)

    for _module_id, em in program.modules.items():
        for node in em.initializers:
            _validate_expr(node, ctx)

    # Cheap-tier param validation (location checks only — deep is in _validate_program_tables).
    if not deep:
        for ir_param in program.params:
            _validate_ir_param(ir_param, ctx)
