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
       functions table.
    3. Each ``program.nominals`` entry: ``descriptor.nominal`` equals its key.
    4. Every ``SymbolId`` referenced by ``IrLoad``/``IrBind``/``IrAssign``
       exists in ``program.symbols``.
    5. The root symbol of every ``IrAssign`` is mutable (``mutable=True``).
    6. Every ``Location`` on every node (and ``IrIndexStep``): its
       ``source_id`` exists in ``program.sources``; and
       ``0 <= start_offset <= end_offset <= len(normalized_text)``.
    7. ``program.functions`` contains every callable descriptor. A reference
       from ``IrMakeClosure``/``IrDirectCall`` (or a symbol owner) must resolve
       there; extern boundary contracts are checked for internal consistency
       (registered nominals, type-variable positions matching their declared
       type parameters).

The expression dispatcher uses a closed structural ``match`` with a final
``assert_never(node)`` arm so that adding an ``IrExpr`` variant in a
future change without a validator arm produces a mypy exhaustiveness error.

``validate_ir`` raises ``InvalidIrError`` on the *first* violation found.
"""

from __future__ import annotations

import decimal
from collections.abc import Callable, Mapping
from typing import TypeVar, assert_never

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
from agm.agl.ir.ids import ContractId, FunctionId, Location, NominalId, SourceId, SymbolId
from agm.agl.ir.nodes import (
    AutoTraceField,
    IrAgentHandle,
    IrAnd,
    IrArith,
    IrAsk,
    IrAskRequest,
    IrAssign,
    IrBind,
    IrBlock,
    IrBreak,
    IrCase,
    IrCaseArm,
    IrCatchHandler,
    IrCoerce,
    IrCompare,
    IrConfigBind,
    IrConstBool,
    IrConstDecimal,
    IrConstInt,
    IrConstJsonNull,
    IrConstText,
    IrConstUnit,
    IrContains,
    IrContinue,
    IrConvert,
    IrDirectCall,
    IrEnumCaseKey,
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
    IrLiteralCaseKey,
    IrLiteralKind,
    IrLoad,
    IrLoop,
    IrMakeClosure,
    IrMakeConstructor,
    IrMakeDict,
    IrMakeEnum,
    IrMakeException,
    IrMakeList,
    IrMakeRecord,
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
    UseDefault,
)
from agm.agl.ir.operations import ArithKind, ArithOp, CmpOp, CompareKind, UnaryOp
from agm.agl.ir.program import (
    ExecutableProgram,
    ExternFunctionBody,
    IrFunctionBody,
    IrParam,
    NominalKind,
    SourceFile,
)
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

    __slots__ = (
        "active_exprs",
        "check_payload_dominance",
        "deep",
        "dominating_payload_symbols",
        "payload_symbols",
        "program",
        "payload_requirements",
        "requirement_collectors",
        "validated_exprs",
    )

    def __init__(
        self,
        program: ExecutableProgram,
        *,
        deep: bool,
        payload_symbols: set[SymbolId],
        check_payload_dominance: bool = False,
    ) -> None:
        self.program = program
        self.deep = deep
        self.check_payload_dominance = check_payload_dominance
        self.payload_symbols = payload_symbols
        self.dominating_payload_symbols: frozenset[SymbolId] = frozenset()
        self.active_exprs: set[int] = set()
        self.validated_exprs: set[int] = set()
        # A cached expression's free payload requirements are independent of
        # its incoming case-arm bindings.  This preserves DAG sharing while
        # still checking that every incoming path supplies those bindings.
        self.payload_requirements: dict[int, frozenset[SymbolId]] = {}
        self.requirement_collectors: list[set[SymbolId]] = []


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
            _check_ref_chain(
                key,
                defs,
                lambda t: t.key if isinstance(t, RefDecode) else None,
                ref_kind="DecodeSchema RefDecode",
                key_noun="$defs key",
            )
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


_RefT = TypeVar("_RefT")


def _check_ref_chain(
    key: str,
    defs: "Mapping[str, _RefT]",
    follow: "Callable[[_RefT], str | None]",
    *,
    ref_kind: str,
    key_noun: str,
) -> None:
    """Ensure a chain of ``$defs`` refs reaches a non-ref body without cycling.

    *follow* returns the next key when its argument is itself a ref node
    (``RefDecode``/``BoundaryRef``), or ``None`` at a concrete body where the
    chain terminates.  A key absent from *defs* or revisited (a cycle) is an
    IR invariant violation; *ref_kind* and *key_noun* name the schema flavour
    and its defs-key wording in the message.
    """
    seen: set[str] = set()
    current = key
    while True:
        if current in seen:
            raise InvalidIrError(f"{ref_kind} cycle reaches no body at {key_noun} {current!r}")
        seen.add(current)
        target = defs.get(current)
        if target is None:
            raise InvalidIrError(f"{ref_kind} references unknown {key_noun} {current!r}")
        next_key = follow(target)
        if next_key is None:
            return
        current = next_key


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
            _check_ref_chain(
                key,
                defs,
                lambda t: t.key if isinstance(t, BoundaryRef) else None,
                ref_kind="BoundarySchema BoundaryRef",
                key_noun="defs key",
            )
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
                f"FunctionDescriptor for {fn_key!r}: contract has duplicate"
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
            f"FunctionDescriptor for {fn_key!r}: contract references type"
            f" variable(s) {sorted(unknown)!r} not declared in type_params"
            f" {contract.type_params!r}"
        )


def _resolve_callable_params(
    fn_id: FunctionId, ctx: _Context, node_desc: str
) -> "tuple[IrFunctionParam, ...]":
    """Resolve *fn_id* to its declared parameter tuple.

    ``function_id``s resolve through the unified ``program.functions`` table.
    """
    fn_desc = ctx.program.functions.get(fn_id)
    if fn_desc is not None:
        return fn_desc.params
    raise InvalidIrError(
        f"{node_desc} references function_id={fn_id!r} which is not in"
        " program.functions"
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
# IrCase validation
# ---------------------------------------------------------------------------


def _literal_key_is_valid(key: IrLiteralCaseKey) -> bool:
    value = key.scalar_value
    return (
        key.kind is IrLiteralKind.NUMERIC
        and isinstance(value, decimal.Decimal)
        and value.is_finite()
        or key.kind is IrLiteralKind.BOOL
        and isinstance(value, bool)
        or key.kind is IrLiteralKind.TEXT
        and isinstance(value, str)
        or key.kind is IrLiteralKind.NULL
        and value is None
    )


def _case_family(arm: IrCaseArm) -> tuple[str, object]:
    if isinstance(arm.key, IrEnumCaseKey):
        return "enum", arm.key.nominal
    return "literal", arm.key.kind


def _validate_case_arm(arm: IrCaseArm, ctx: _Context) -> None:
    match arm.key:
        case IrEnumCaseKey(nominal=nominal, variant=variant):
            if ctx.deep:
                descriptor = ctx.program.nominals.get(nominal)
                if descriptor is None:
                    raise InvalidIrError(
                        f"IrEnumCaseKey references nominal {nominal!r}"
                        " which is not in program.nominals"
                    )
                if descriptor.kind is not NominalKind.ENUM:
                    raise InvalidIrError(
                        f"IrEnumCaseKey references non-enum nominal {nominal!r}"
                    )
                variant_descriptor = next(
                    (item for item in descriptor.variants if item.name == variant), None
                )
                if variant_descriptor is None:
                    raise InvalidIrError(
                        f"IrEnumCaseKey references unknown variant {variant!r}"
                        f" of nominal {nominal!r}"
                    )
                valid_fields = set(variant_descriptor.fields)
            else:
                valid_fields = None
        case IrLiteralCaseKey() as key:
            if not _literal_key_is_valid(key):
                raise InvalidIrError(f"IrLiteralCaseKey has invalid scalar {key.scalar_value!r}")
            if arm.field_bindings:
                raise InvalidIrError("literal IrCaseArm must not bind payload fields")
            valid_fields = None
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)

    field_names: set[str] = set()
    binding_symbols: set[SymbolId] = set()
    for field_name, symbol in arm.field_bindings:
        if field_name in field_names:
            raise InvalidIrError(f"IrCaseArm binds field {field_name!r} more than once")
        field_names.add(field_name)
        if symbol in binding_symbols:
            raise InvalidIrError(
                f"IrCaseArm binds destination symbol {symbol.value!r} more than once"
            )
        binding_symbols.add(symbol)
        if valid_fields is not None and field_name not in valid_fields:
            raise InvalidIrError(
                f"IrCaseArm binds unknown immediate field {field_name!r}"
            )
        if ctx.deep:
            symbol_descriptor = ctx.program.symbols.get(symbol)
            if symbol_descriptor is None:
                raise InvalidIrError(
                    f"IrCaseArm field binding references unknown symbol_id={symbol.value!r}"
                )
            if (
                symbol_descriptor.mutable
                or symbol_descriptor.public_name is not None
                or not symbol_descriptor.synthetic
            ):
                raise InvalidIrError(
                    "IrCaseArm field binding symbols must be private immutable "
                    "synthetic temporaries"
                )


def _validate_case(node: IrCase, ctx: _Context) -> None:
    _validate_location(node.location, ctx)
    _validate_expr(node.subject, ctx)
    seen_keys: set[object] = set()
    family: tuple[str, object] | None = None
    for arm in node.arms:
        arm_family = _case_family(arm)
        if family is None:
            family = arm_family
        elif arm_family != family:
            raise InvalidIrError("IrCase arms use incompatible discriminant families")
        _validate_case_arm(arm, ctx)
        ctx.payload_symbols.update(symbol for _field_name, symbol in arm.field_bindings)
        prior_payload_symbols = ctx.dominating_payload_symbols
        ctx.dominating_payload_symbols = prior_payload_symbols | frozenset(
            symbol for _field_name, symbol in arm.field_bindings
        )
        try:
            requirements = _validate_expr(arm.body, ctx, merge_requirements=False)
            ctx.requirement_collectors[-1].update(
                requirements - frozenset(symbol for _field_name, symbol in arm.field_bindings)
            )
        finally:
            ctx.dominating_payload_symbols = prior_payload_symbols
        if arm.key in seen_keys:
            raise InvalidIrError(f"IrCase contains duplicate runtime key {arm.key!r}")
        seen_keys.add(arm.key)
    if node.default is not None:
        _validate_expr(node.default, ctx)
    elif family == ("literal", IrLiteralKind.BOOL):
        bool_keys = {
            arm.key.scalar_value
            for arm in node.arms
            if isinstance(arm.key, IrLiteralCaseKey)
        }
        if bool_keys != {True, False}:
            raise InvalidIrError("IrCase has an incomplete boolean domain without a default")
    elif family is not None and family[0] == "enum" and ctx.deep:
        nominal = family[1]
        assert isinstance(nominal, NominalId)
        descriptor = ctx.program.nominals[nominal]
        enum_keys = {
            arm.key.variant for arm in node.arms if isinstance(arm.key, IrEnumCaseKey)
        }
        if enum_keys != {variant.name for variant in descriptor.variants}:
            raise InvalidIrError("IrCase has an incomplete enum domain without a default")
    elif family is None or family[0] == "literal":
        raise InvalidIrError("IrCase over an open domain requires a default")


# ---------------------------------------------------------------------------
# Closed-union expression dispatcher
# ---------------------------------------------------------------------------


def _validate_expr(
    node: IrExpr, ctx: _Context, *, merge_requirements: bool = True
) -> frozenset[SymbolId]:
    """Validate a DAG node once and check its free payload requirements."""
    identifier = id(node)
    if identifier in ctx.active_exprs:
        raise InvalidIrError("IR expression graph contains a cycle")

    requirements = ctx.payload_requirements.get(identifier)
    if requirements is None:
        ctx.active_exprs.add(identifier)
        ctx.requirement_collectors.append(set())
        try:
            _validate_expr_node(node, ctx)
            requirements = frozenset(ctx.requirement_collectors[-1])
        finally:
            ctx.requirement_collectors.pop()
            ctx.active_exprs.remove(identifier)
        ctx.validated_exprs.add(identifier)
        ctx.payload_requirements[identifier] = requirements

    if ctx.check_payload_dominance:
        missing = requirements - ctx.dominating_payload_symbols
        if missing:
            symbol = next(iter(missing))
            raise InvalidIrError(
                f"IrLoad references payload symbol_id={symbol.value!r} outside a binding IrCaseArm"
            )
    if merge_requirements and ctx.requirement_collectors:
        ctx.requirement_collectors[-1].update(requirements)
    return requirements


def _validate_expr_node(node: IrExpr, ctx: _Context) -> None:
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
                if ctx.check_payload_dominance and node.symbol in ctx.payload_symbols:
                    ctx.requirement_collectors[-1].add(node.symbol)

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

        case IrCase():
            _validate_case(node, ctx)

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


def _validate_program_tables(
    program: ExecutableProgram,
    payload_symbols: set[SymbolId],
    *,
    check_payload_dominance: bool = False,
) -> None:
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
    fn_ctx = _Context(
        program,
        deep=True,
        payload_symbols=payload_symbols,
        check_payload_dominance=check_payload_dominance,
    )
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
        match fn_desc.impl:
            case IrFunctionBody(body=body):
                _validate_expr(body, fn_ctx)
            case ExternFunctionBody(contract=contract):
                if len(fn_desc.params) != len(contract.params):
                    raise InvalidIrError(
                        f"FunctionDescriptor for {fn_key!r} has {len(fn_desc.params)}"
                        f" IR params but its contract has {len(contract.params)}"
                        " boundary params"
                    )
                _validate_extern_contract(fn_key, contract, fn_ctx)
            case other:  # pragma: no cover
                assert_never(other)

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
    payload_symbols: set[SymbolId] = set()
    ctx = _Context(program, deep=deep, payload_symbols=payload_symbols)

    if deep:
        _validate_program_tables(program, payload_symbols)

    for _module_id, em in program.modules.items():
        for node in em.initializers:
            _validate_expr(node, ctx)

    if deep:
        # The first traversal inventories every field-binding symbol. Re-run
        # with that complete inventory so shared DAG nodes are checked on every
        # payload-binding path, independent of traversal order.
        _validate_program_tables(
            program, payload_symbols, check_payload_dominance=True
        )
        dominance_ctx = _Context(
            program,
            deep=True,
            payload_symbols=payload_symbols,
            check_payload_dominance=True,
        )
        for _module_id, em in program.modules.items():
            for node in em.initializers:
                _validate_expr(node, dominance_ctx)

    # Cheap-tier param validation (location checks only — deep is in _validate_program_tables).
    if not deep:
        for ir_param in program.params:
            _validate_ir_param(ir_param, ctx)
