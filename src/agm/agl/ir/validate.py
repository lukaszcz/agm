"""Structural IR validator for the AgL typeless execution IR (M1-C).

Two tiers (D6 — validate_ir runs ONLY when explicitly called):

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
       ``program.modules``; a ``FunctionId`` owner is a violation in M1
       (no functions table yet).
    3. Each ``program.nominals`` entry: ``descriptor.nominal`` equals its key.
    4. Every ``SymbolId`` referenced by ``IrLoad``/``IrBind``/``IrAssign``
       exists in ``program.symbols``.
    5. The root symbol of every ``IrAssign`` is mutable (``mutable=True``).
    6. Every ``Location`` on every node (and ``IrIndexStep``): its
       ``source_id`` exists in ``program.sources``; and
       ``0 <= start_offset <= end_offset <= len(normalized_text)``.

The expression dispatcher uses a closed structural ``match`` with a final
``assert_never(node)`` arm (D4) so that adding an ``IrExpr`` variant in a
later milestone without a validator arm produces a mypy exhaustiveness error.

``validate_ir`` raises ``InvalidIrError`` on the *first* violation found.
"""

from __future__ import annotations

from typing import assert_never

from agm.agl.ir.contracts import (
    ConversionStrategy,
    DecodeSchema,
    DictDecode,
    EnumDecode,
    ListDecode,
    RecordDecode,
    ScalarDecode,
)
from agm.agl.ir.ids import FunctionId, Location, NominalId, SourceId
from agm.agl.ir.nodes import (
    AutoTraceField,
    IrAnd,
    IrArith,
    IrAssign,
    IrBind,
    IrBindPlan,
    IrBlock,
    IrCase,
    IrCatchHandler,
    IrCoerce,
    IrCompare,
    IrConstBool,
    IrConstDecimal,
    IrConstInt,
    IrConstJsonNull,
    IrConstructorPlan,
    IrConstText,
    IrConstUnit,
    IrContains,
    IrConvert,
    IrDirectCall,
    IrExpr,
    IrField,
    IrIf,
    IrIndex,
    IrIndexStep,
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
    IrRaise,
    IrRenderTemplate,
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
from agm.agl.ir.program import ExecutableProgram, NominalKind, SourceFile
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
# Nominal completeness check helpers (deep tier, M3d)
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
) -> None:
    """Enforce that decode strategies carry schema+decode and total strategies do not."""
    needs_decode = strategy in _DECODE_STRATEGIES
    has_decode = json_schema is not None and decode is not None
    if needs_decode and not has_decode:
        raise InvalidIrError(
            f"ConversionRecipe strategy {strategy.value!r} requires json_schema and decode"
        )
    if not needs_decode and (json_schema is not None or decode is not None):
        raise InvalidIrError(
            f"ConversionRecipe strategy {strategy.value!r} must not carry json_schema/decode"
        )


def _check_decode_nominals(decode: DecodeSchema, ctx: _Context) -> None:
    """Deep tier: every nominal referenced by a decode schema must be registered."""
    match decode:
        case ScalarDecode():
            return
        case ListDecode(elem=elem):
            _check_decode_nominals(elem, ctx)
        case DictDecode(value=value_schema):
            _check_decode_nominals(value_schema, ctx)
        case RecordDecode(nominal=nominal, fields=fields):
            _check_nominal_in_table(nominal, ctx)
            for _fname, fschema in fields:
                _check_decode_nominals(fschema, ctx)
        case EnumDecode(nominal=nominal, variants=variants):
            _check_nominal_in_table(nominal, ctx)
            for variant in variants:
                for _fname, fschema in variant.fields:
                    _check_decode_nominals(fschema, ctx)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


# ---------------------------------------------------------------------------
# IrIndexStep validation
# ---------------------------------------------------------------------------


def _validate_index_step(step: IrIndexStep, ctx: _Context) -> None:
    _validate_location(step.location, ctx)
    _validate_expr(step.index, ctx)


# ---------------------------------------------------------------------------
# IrCatchHandler validation (deep tier, M3f-A)
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
# IrMatchPlan validator (closed union, D4)
# ---------------------------------------------------------------------------


def _validate_match_plan(plan: IrMatchPlan, ctx: _Context) -> None:
    """Validate a closed ``IrMatchPlan`` node recursively.

    The final ``assert_never`` arm (D4) ensures mypy reports a type error when
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
# Closed-union expression dispatcher (D4)
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
            _check_recipe_consistency(recipe.strategy, recipe.json_schema, recipe.decode)
            if ctx.deep and recipe.decode is not None:
                _check_decode_nominals(recipe.decode, ctx)
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

        case IrLoop(limit=limit, body=body, condition=condition):
            _validate_location(node.location, ctx)
            if limit is not None and limit < 0:
                raise InvalidIrError(
                    f"IrLoop has limit={limit!r} which is negative (must be >= 0 or None)"
                )
            _validate_expr(body, ctx)
            _validate_expr(condition, ctx)

        case IrMakeClosure(function_id=fn_id, captures=captures):
            _validate_location(node.location, ctx)
            if ctx.deep:
                if fn_id not in ctx.program.functions:
                    raise InvalidIrError(
                        f"IrMakeClosure references function_id={fn_id!r}"
                        " which is not in program.functions"
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
                if fn_id not in ctx.program.functions:
                    raise InvalidIrError(
                        f"IrDirectCall references function_id={fn_id!r}"
                        " which is not in program.functions"
                    )
                fn_desc = ctx.program.functions[fn_id]
                if len(arguments) != len(fn_desc.params):
                    raise InvalidIrError(
                        f"IrDirectCall to function_id={fn_id!r} has {len(arguments)}"
                        f" arguments but the function has {len(fn_desc.params)} parameters"
                    )
                for index, arg in enumerate(arguments):
                    if isinstance(arg, UseDefault):
                        if arg.param_index != index:
                            raise InvalidIrError(
                                f"IrDirectCall to function_id={fn_id!r}: UseDefault at"
                                f" position {index} has param_index={arg.param_index}"
                                " (must equal its position)"
                            )
                        if fn_desc.params[index].default is None:
                            raise InvalidIrError(
                                f"IrDirectCall to function_id={fn_id!r}: UseDefault for"
                                f" parameter {index} which has no default"
                            )
            for arg in arguments:
                if not isinstance(arg, UseDefault):
                    _validate_expr(arg, ctx)

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
            if owner not in program.functions:
                raise InvalidIrError(
                    f"SymbolDescriptor for symbol_id={sym_key!r} has owner={owner!r}"
                    " which is not in program.functions"
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

    # (Sources table has no key/id consistency invariant beyond being keyed by
    # SourceId; key consistency is structural to dict construction.)


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
