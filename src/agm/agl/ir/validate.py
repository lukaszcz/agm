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
       An EXCEPTION descriptor's ``base`` is ``None`` or an EXCEPTION
       descriptor present in the table, and the base chain is acyclic;
       RECORD and ENUM descriptors have ``base is None``.
    4. Every ``SymbolId`` referenced by ``IrLoad``/``IrBind``/``IrAssign``
       exists in ``program.symbols``.
    5. The root symbol of every ``IrAssign`` is mutable (``mutable=True``).
    6. Every ``Location`` on every node: its
       ``source_id`` exists in ``program.sources``; and
       ``0 <= start_offset <= end_offset <= len(normalized_text)``.
    7. ``program.functions`` contains every callable descriptor. A reference
       from ``IrMakeClosure``/``IrDirectCall`` (or a symbol owner) must resolve
       there; extern boundary contracts are checked for internal consistency
       (registered nominals, type-variable positions matching their declared
       type parameters).
    8. Every non-engine module-qualified ``IrBuiltinLoad``/``IrBuiltinStore``
       and declared builtin-default key identifies a loaded host-backed binding
       declaration, including its scope path. Canonical root ``std/config``
       engine keys (including legacy strings) remain valid independently.
    9. Every ``IrField`` nominal is registered, field-bearing, and declares
       its projected field (on at least one enum payload shape for enums).
       Every ``IrFieldSet`` targets a declared mutable field of a registered
       record nominal.
    10. ``program_symbols`` and ``program_functions`` form a one-to-one, bidirectional index of
        linked ``program def`` entries with registered symbols and ``IrFunctionBody`` functions;
        when present, ``synthetic_main_symbol`` resolves through ``program_functions`` to exactly
        one marked synthetic ``FunctionDescriptor``.
    11. ``program_signatures`` is keyed by exactly the ``program_functions``
        symbols, and each entry's parameter count and required-ness agree
        with its function descriptor's own declared parameters.
    12. ``program_configs`` is sparse over ``program_functions`` symbols, with
        no duplicate target per entry, each ``std/config``-rooted target names
        a known engine setting, and each value expression validates.

The expression dispatcher uses a closed structural ``match`` with a final
``assert_never(node)`` arm so that adding an ``IrExpr`` variant in a
future change without a validator arm produces a mypy exhaustiveness error.

``validate_ir`` raises ``InvalidIrError`` on the *first* violation found.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar, assert_never

from agm.agl.ir.builtin_vars import BuiltinVarKey, is_engine_builtin_var_key
from agm.agl.ir.contracts import (
    ArrayDecode,
    ArrayEncode,
    ContractRequest,
    ConversionStrategy,
    DecodeSchema,
    DictDecode,
    DictEncode,
    EncodeDefinition,
    EncodeSchema,
    EnumDecode,
    EnumEncode,
    ExceptionEncode,
    FieldDecode,
    FieldEncode,
    RecordDecode,
    RecordEncode,
    RefDecode,
    RefEncode,
    ScalarDecode,
    ScalarEncode,
    TypeParameterEncode,
    forwarded_encode_key,
)
from agm.agl.ir.ids import ContractId, FunctionId, Location, NominalId, SourceId, SymbolId
from agm.agl.ir.nodes import (
    IrAnd,
    IrArith,
    IrAsk,
    IrAskRequest,
    IrAssign,
    IrBind,
    IrBlock,
    IrBreak,
    IrBuiltinLoad,
    IrBuiltinStore,
    IrCase,
    IrCaseArm,
    IrCatchHandler,
    IrCoerce,
    IrCompare,
    IrConstBool,
    IrConstDecimal,
    IrConstInt,
    IrConstJsonNull,
    IrConstText,
    IrConstUnit,
    IrContains,
    IrContinue,
    IrConvert,
    IrCopyValue,
    IrDirectCall,
    IrExec,
    IrExpr,
    IrField,
    IrFieldMode,
    IrFieldSet,
    IrFunctionParam,
    IrIf,
    IrIndex,
    IrIndexSet,
    IrIndirectCall,
    IrIterHasNext,
    IrIterInit,
    IrIterNext,
    IrLiteralCaseKey,
    IrLiteralKind,
    IrLoad,
    IrLoop,
    IrMakeArray,
    IrMakeClosure,
    IrMakeConstructor,
    IrMakeDict,
    IrMakeException,
    IrMakeJsonArray,
    IrMakeJsonObject,
    IrMakeRecord,
    IrNominalCaseKey,
    IrNominalCast,
    IrNominalIs,
    IrOr,
    IrPrint,
    IrRaise,
    IrRenderTemplate,
    IrRenderValue,
    IrResource,
    IrReturn,
    IrSequence,
    IrSessionAsk,
    IrSessionDefault,
    IrSessionOp,
    IrSessionOpen,
    IrSessionOpKind,
    IrTemplateText,
    IrTemplateValue,
    IrTry,
    IrUnary,
    IrUpdateRecord,
    UseDefault,
    is_canonical_literal_scalar,
)
from agm.agl.ir.operations import ArithKind, ArithOp, CmpOp, CompareKind, UnaryOp
from agm.agl.ir.program import (
    ExecutableProgram,
    ExternFunctionBody,
    IrFunctionBody,
    IrProgramParam,
    NominalKind,
    SourceFile,
)
from agm.agl.ir.static_keys import StaticBindingKey
from agm.agl.modules.ids import STD_CONFIG_ID, ModuleId
from agm.config.engine_keys import ENGINE_KEY_NAMES

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
        "dominators",
        "payload_symbols",
        "program",
        "payload_requirements",
        "requirement_collectors",
    )

    def __init__(
        self,
        program: ExecutableProgram,
        *,
        deep: bool,
        check_payload_dominance: bool = False,
    ) -> None:
        self.program = program
        self.deep = deep
        self.check_payload_dominance = check_payload_dominance
        # Every symbol bound by an IrCase arm, inventoried as the traversal
        # meets each arm; complete once the traversal finishes.
        self.payload_symbols: set[SymbolId] = set()
        self.dominating_payload_symbols: frozenset[SymbolId] = frozenset()
        self.active_exprs: set[int] = set()
        # A cached expression's free payload requirements are independent of
        # its incoming case-arm bindings.  This preserves DAG sharing while
        # still checking that every incoming path supplies those bindings.
        self.payload_requirements: dict[int, frozenset[SymbolId]] = {}
        # Per visited expression, the intersection of the payload symbols bound
        # on every path that reaches it — the symbols it may rely on.
        self.dominators: dict[int, frozenset[SymbolId]] = {}
        self.requirement_collectors: list[set[SymbolId]] = []


# ---------------------------------------------------------------------------
# Location validation helpers
# ---------------------------------------------------------------------------


def _check_location_cheap(loc: Location) -> None:
    """Validate local structural invariants on a ``Location``."""
    if loc.start_offset < 0:
        raise InvalidIrError(f"Location has negative start_offset={loc.start_offset!r}")
    if loc.start_offset > loc.end_offset:
        raise InvalidIrError(
            f"Location has start_offset={loc.start_offset!r} > end_offset={loc.end_offset!r}"
        )
    if loc.start_line < 1:
        raise InvalidIrError(f"Location has start_line={loc.start_line!r} (must be >= 1)")
    if loc.start_col < 0:
        raise InvalidIrError(f"Location has negative start_col={loc.start_col!r}")


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


def _validate_builtin_key(key: BuiltinVarKey | str, ctx: _Context) -> None:
    """Validate a host-backed binding key, including legacy engine strings."""
    if isinstance(key, str):
        if not ctx.deep:
            return
        if key not in ENGINE_KEY_NAMES:
            raise InvalidIrError(f"unknown engine builtin-var key {key!r}")
        return
    module_id, scope_path, name = key
    if not ctx.deep:
        return
    if is_engine_builtin_var_key(key):
        return
    if module_id == STD_CONFIG_ID and not scope_path:
        raise InvalidIrError(f"IR builtin-var node has unknown engine key {name!r}")
    if module_id not in ctx.program.modules:
        raise InvalidIrError(f"IR builtin-var key references unloaded module {module_id!r}")
    if key not in ctx.program.builtin_var_declarations:
        raise InvalidIrError(f"IR builtin-var key has no matching declaration: {key!r}")


# ---------------------------------------------------------------------------
# Nominal completeness check helpers (deep tier)
# ---------------------------------------------------------------------------


def _check_nominal_in_table(nominal: NominalId, ctx: _Context) -> None:
    """Raise ``InvalidIrError`` if *nominal* is not in ``program.nominals``."""
    if nominal not in ctx.program.nominals:
        raise InvalidIrError(
            f"IR node references nominal {nominal!r} which is not in program.nominals"
        )


def _validate_use_default_slot(
    context: str,
    noun: str,
    value: UseDefault,
    index: int,
    ctx: _Context,
    has_default: "bool | None",
) -> None:
    """Validate one ``UseDefault`` sentinel: shared by call arguments and
    constructor field slots.

    The sentinel's ``param_index`` must equal its list position -- a
    structural, node-local invariant with no program-table dependency, so it
    is checked unconditionally, in both tiers. Whether the referenced slot
    actually carries a default requires the callee's/nominal's resolved
    parameter table, so it is checked only under ``ctx.deep`` and only when
    *has_default* is supplied (``None`` when the caller has no such table to
    resolve, e.g. an unresolved ``function_id``/nominal under the cheap tier).
    *context* is prose already naming the call/construction (e.g.
    ``"IrDirectCall to function_id=3"``); *noun* is ``"parameter"`` or
    ``"field"``.
    """
    if value.param_index != index:
        raise InvalidIrError(
            f"{context}: UseDefault at position {index} has param_index="
            f"{value.param_index} (must equal its position)"
        )
    if ctx.deep and has_default is False:
        raise InvalidIrError(f"{context}: UseDefault for {noun} {index} which has no default")


def _validate_constructor_fields(
    nominal: NominalId,
    fields: "tuple[tuple[str, IrExpr | UseDefault], ...]",
    ctx: _Context,
    node_name: str,
) -> None:
    """Validate an ``IrMakeRecord``/``IrMakeException``'s field slots.

    Mirrors ``IrDirectCall``'s argument checks: field-slot count and name
    parity against the constructed nominal's own declared fields, then each
    ``UseDefault`` slot through :func:`_validate_use_default_slot` (deep tier
    only, since both need the resolved ``NominalDescriptor``). Every other
    slot is a plain expression.
    """
    if ctx.deep:
        _check_nominal_in_table(nominal, ctx)
    desc = ctx.program.nominals.get(nominal)
    context = f"{node_name} for nominal {nominal!r}"
    if ctx.deep and desc is not None:
        if len(fields) != len(desc.fields):
            raise InvalidIrError(
                f"{context} has {len(fields)} field slots but the nominal declares"
                f" {len(desc.fields)} fields"
            )
        for index, (fname, _fexpr) in enumerate(fields):
            if fname != desc.fields[index]:
                raise InvalidIrError(
                    f"{context}: field slot {index} is named {fname!r} but the nominal"
                    f" declares {desc.fields[index]!r} at that position"
                )
    for index, (_fname, fexpr) in enumerate(fields):
        if isinstance(fexpr, UseDefault):
            _validate_use_default_slot(
                context,
                "field",
                fexpr,
                index,
                ctx,
                desc.field_defaults[index] is not None if ctx.deep and desc is not None else None,
            )
        else:
            _validate_expr(fexpr, ctx)


def _check_nominal_field(
    nominal: NominalId, field: str, mode: IrFieldMode, ctx: _Context, node_name: str
) -> None:
    """Reject a field reference outside the declaring nominal's field shape.

    Exact and upper-bound projections use the same declaration descriptor for
    field existence. The mode changes runtime identity checking, not which
    fields the statically selected nominal declares.
    """
    desc = ctx.program.nominals.get(nominal)
    if desc is None:  # pragma: no cover
        return  # already caught by _check_nominal_in_table
    match mode:
        case IrFieldMode.EXACT | IrFieldMode.UPPER_BOUND:
            pass
        case _ as _unreachable_mode:  # pragma: no cover
            assert_never(_unreachable_mode)
    if desc.kind is NominalKind.ENUM:
        known_fields = {name for variant in desc.variants for name in variant.fields}
    else:
        known_fields = set(desc.fields)
    if field not in known_fields:
        raise InvalidIrError(
            f"{node_name} references unknown field {field!r} of nominal {nominal!r}"
        )


def _check_record_nominal(nominal: NominalId, ctx: _Context, node_name: str) -> None:
    """Require a nominal-dispatch target to be a member-record identity."""
    _check_nominal_in_table(nominal, ctx)
    if ctx.program.nominals[nominal].kind is not NominalKind.RECORD:
        raise InvalidIrError(f"{node_name} references non-record nominal {nominal!r}")


def _check_downcast_nominals(
    nominals: tuple[NominalId, ...], ctx: _Context, node_name: str
) -> None:
    """Require a downcast's accepted set: all RECORD, or exactly one EXCEPTION."""
    for nominal in nominals:
        _check_nominal_in_table(nominal, ctx)
    kinds = {ctx.program.nominals[n].kind for n in nominals}
    if kinds == {NominalKind.RECORD}:
        return
    if kinds == {NominalKind.EXCEPTION} and len(nominals) == 1:
        return
    raise InvalidIrError(
        f"{node_name} nominals {nominals!r} must be RECORD members or exactly one EXCEPTION"
    )


def _check_nominal_is_target(nominal: NominalId, ctx: _Context, node_name: str) -> None:
    """Require an `is`-test target to be a record or exception identity."""
    _check_nominal_in_table(nominal, ctx)
    if ctx.program.nominals[nominal].kind not in (NominalKind.RECORD, NominalKind.EXCEPTION):
        raise InvalidIrError(f"{node_name} references non-record/non-exception nominal {nominal!r}")


def _check_mutable_record_field(nominal: NominalId, field: str, ctx: _Context) -> None:
    """Require a mutable field on the precise record declaration for a store."""
    _check_record_nominal(nominal, ctx, "IrFieldSet")
    _check_nominal_field(nominal, field, IrFieldMode.EXACT, ctx, "IrFieldSet")
    if field not in ctx.program.nominals[nominal].mutable_fields:
        raise InvalidIrError(
            f"IrFieldSet references immutable field {field!r} of nominal {nominal!r}"
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
    encode: EncodeSchema | None,
    encode_definitions: "tuple[EncodeDefinition, ...]",
) -> None:
    """Require only the conversion metadata selected by each strategy."""
    needs_decode = strategy in _DECODE_STRATEGIES
    has_decode = json_schema is not None and decode is not None
    if needs_decode and not has_decode:
        raise InvalidIrError(
            f"ConversionRecipe strategy {strategy.value!r} requires json_schema and decode"
        )
    if not needs_decode and (json_schema is not None or decode is not None or defs):
        raise InvalidIrError(
            f"ConversionRecipe strategy {strategy.value!r} must not carry json_schema/decode/defs"
        )
    needs_encode = strategy is ConversionStrategy.TO_JSON
    if needs_encode and encode is None:
        raise InvalidIrError("ConversionRecipe strategy 'to_json' requires encode")
    if not needs_encode and (encode is not None or encode_definitions):
        raise InvalidIrError(
            f"ConversionRecipe strategy {strategy.value!r} must not carry encode/encode_definitions"
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


class _EncodeWalk:
    """The definition table one encode-plan walk resolves against, and what it reached.

    Reachability is accumulated across the whole plan rather than per body: a
    definition is legitimate only if some reference — from the root or from
    another reached definition — names it.
    """

    __slots__ = ("ctx", "definitions", "reached")

    def __init__(self, definitions: "Mapping[str, EncodeDefinition]", ctx: _Context) -> None:
        self.definitions = definitions
        self.ctx = ctx
        self.reached: set[str] = set()


def _check_encode_nominals(
    encode: EncodeSchema, definitions: "tuple[EncodeDefinition, ...]", ctx: _Context
) -> None:
    """Validate the root and every reachable definition body of one encode plan.

    Each definition is walked under its own arity, so a
    ``TypeParameterEncode`` inside it is checked against the parameters it can
    actually bind; the root binds none. A definition no reference reaches is
    rejected rather than validated in isolation.
    """
    by_key: dict[str, EncodeDefinition] = {}
    for definition in definitions:
        if definition.key in by_key:
            raise InvalidIrError(f"EncodeSchema has duplicate $defs key {definition.key!r}")
        if definition.parameter_count < 0:
            raise InvalidIrError(
                f"EncodeDefinition {definition.key!r} has a negative parameter count"
            )
        by_key[definition.key] = definition
    walk = _EncodeWalk(by_key, ctx)
    _walk_encode_schema(encode, walk, 0)
    pending = list(walk.reached)
    while pending:
        definition = by_key[pending.pop()]
        before = set(walk.reached)
        _walk_encode_schema(definition.body, walk, definition.parameter_count)
        pending.extend(walk.reached - before)
    if walk.reached != set(by_key):
        raise InvalidIrError("EncodePlan has unreachable definitions")


def _walk_encode_schema(encode: EncodeSchema, walk: _EncodeWalk, parameter_count: int) -> None:
    """Walk one encode-schema node without expanding the definitions it references."""
    ctx = walk.ctx
    match encode:
        case ScalarEncode():
            return
        case TypeParameterEncode(index=index):
            if index < 0 or index >= parameter_count:
                raise InvalidIrError(
                    f"TypeParameterEncode index {index} is outside its definition arity"
                )
        case RefEncode(key=key, arguments=arguments):
            target = _check_ref_chain(
                key,
                walk.definitions,
                forwarded_encode_key,
                ref_kind="EncodeSchema RefEncode",
                key_noun="$defs key",
            )
            if len(arguments) != target.parameter_count:
                raise InvalidIrError(
                    f"RefEncode supplies {len(arguments)} arguments for the"
                    f" {target.parameter_count} parameters of $defs key {key!r}"
                )
            walk.reached.add(key)
            for argument in arguments:
                _walk_encode_schema(argument, walk, parameter_count)
        case ArrayEncode(elem=elem):
            _walk_encode_schema(elem, walk, parameter_count)
        case DictEncode(value=value_schema):
            _walk_encode_schema(value_schema, walk, parameter_count)
        case RecordEncode(nominal=nominal, fields=fields):
            _check_nominal_in_table(nominal, ctx)
            desc = ctx.program.nominals[nominal]
            if desc.kind is not NominalKind.RECORD:
                raise InvalidIrError(f"RecordEncode references non-record nominal {nominal!r}")
            _check_nominal_fields(fields, desc.fields, "RecordEncode")
            for fenc in fields:
                _walk_encode_schema(fenc.schema, walk, parameter_count)
        case ExceptionEncode(nominal=nominal, fields=fields):
            _check_nominal_in_table(nominal, ctx)
            desc = ctx.program.nominals[nominal]
            if desc.kind is not NominalKind.EXCEPTION:
                raise InvalidIrError(
                    f"ExceptionEncode references non-exception nominal {nominal!r}"
                )
            _check_nominal_fields(fields, desc.fields, "ExceptionEncode")
            for fenc in fields:
                _walk_encode_schema(fenc.schema, walk, parameter_count)
        case EnumEncode(nominal=nominal, variants=variants):
            _check_nominal_in_table(nominal, ctx)
            desc = ctx.program.nominals[nominal]
            if desc.kind is not NominalKind.ENUM:
                raise InvalidIrError(f"EnumEncode references non-enum nominal {nominal!r}")
            if len(variants) != len(desc.variants):
                raise InvalidIrError(f"EnumEncode variants disagree with enum nominal {nominal!r}")
            for variant, expected in zip(variants, desc.variants, strict=True):
                if variant.name != expected.name or variant.nominal != expected.member:
                    raise InvalidIrError(
                        f"EnumEncode variant {variant.name!r} disagrees with"
                        f" enum nominal {nominal!r}"
                    )
                _check_nominal_fields(variant.fields, expected.fields, "EnumEncode variant")
                for fenc in variant.fields:
                    _walk_encode_schema(fenc.schema, walk, parameter_count)
        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


def _check_nominal_fields(
    fields: "tuple[FieldDecode, ...] | tuple[FieldEncode, ...]",
    expected: tuple[str, ...],
    owner: str,
) -> None:
    """Require an encoder or decoder to select exactly its linked declaration's own fields."""
    if tuple(f.name for f in fields) != expected:
        raise InvalidIrError(f"{owner} fields disagree with its nominal descriptor")


def _check_field_decode_defaults(
    fields: "tuple[FieldDecode, ...]", field_defaults: "tuple[object | None, ...]", owner: str
) -> None:
    """Require each ``FieldDecode.default_index`` to agree with its declaring descriptor.

    Mirrors :func:`_validate_use_default_slot`'s construction-side check on the
    decode side: a set ``default_index`` must equal the field's own position,
    and the descriptor's ``field_defaults`` entry at that index must not be
    ``None``.
    """
    for index, fdec in enumerate(fields):
        if fdec.default_index is None:
            continue
        if fdec.default_index != index:
            raise InvalidIrError(
                f"{owner} field {fdec.name!r} has default_index={fdec.default_index}"
                f" (must equal its position {index})"
            )
        if field_defaults[index] is None:
            raise InvalidIrError(
                f"{owner} field {fdec.name!r} has default_index set but its nominal"
                " declares no default at that position"
            )


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
        case ArrayDecode(elem=elem):
            _walk_decode_schema(elem, defs, ctx)
        case DictDecode(value=value_schema):
            _walk_decode_schema(value_schema, defs, ctx)
        case RecordDecode(nominal=nominal, display_name=display_name, fields=fields, name=name):
            _check_nominal_in_table(nominal, ctx)
            record = ctx.program.nominals[nominal]
            if record.kind is not NominalKind.RECORD:
                raise InvalidIrError(f"RecordDecode references non-record nominal {nominal!r}")
            if display_name != record.display_name:
                raise InvalidIrError(
                    f"RecordDecode display name disagrees with nominal {nominal!r}"
                )
            if name != record.declared_name:
                raise InvalidIrError(f"RecordDecode name disagrees with nominal {nominal!r}")
            _check_nominal_fields(fields, record.fields, "RecordDecode")
            _check_field_decode_defaults(fields, record.field_defaults, "RecordDecode")
            for rdec in fields:
                _walk_decode_schema(rdec.schema, defs, ctx)
        case EnumDecode(nominal=nominal, display_name=display_name, variants=variants, name=name):
            _check_nominal_in_table(nominal, ctx)
            enum = ctx.program.nominals[nominal]
            if enum.kind is not NominalKind.ENUM:
                raise InvalidIrError(f"EnumDecode references non-enum nominal {nominal!r}")
            if display_name != enum.display_name or len(variants) != len(enum.variants):
                raise InvalidIrError(f"EnumDecode disagrees with enum nominal {nominal!r}")
            if name != enum.declared_name:
                raise InvalidIrError(f"EnumDecode name disagrees with nominal {nominal!r}")
            for variant, expected in zip(variants, enum.variants, strict=True):
                if variant.name != expected.name or variant.nominal != expected.member:
                    raise InvalidIrError(
                        f"EnumDecode variant {variant.name!r} disagrees with"
                        f" enum nominal {nominal!r}"
                    )
                _check_nominal_in_table(variant.nominal, ctx)
                member = ctx.program.nominals[variant.nominal]
                if variant.display_name != member.display_name:
                    raise InvalidIrError(
                        f"EnumDecode variant {variant.name!r} display name disagrees with"
                        f" member nominal {variant.nominal!r}"
                    )
                if variant.name != member.declared_name:
                    raise InvalidIrError(
                        f"EnumDecode variant {variant.name!r} name disagrees with"
                        f" member nominal {variant.nominal!r}"
                    )
                _check_nominal_fields(variant.fields, expected.fields, "EnumDecode variant")
                _check_field_decode_defaults(
                    variant.fields, member.field_defaults, "EnumDecode variant"
                )
                for vdec in variant.fields:
                    _walk_decode_schema(vdec.schema, defs, ctx)
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
) -> "_RefT":
    """Resolve a chain of ``$defs`` refs to the non-ref body it must reach.

    *follow* returns the next key when its argument is itself a ref node
    (``RefDecode``), or ``None`` at a concrete body where the
    chain terminates.  A key absent from *defs* or revisited (a cycle) is an
    IR invariant violation; *ref_kind* and *key_noun* name the schema flavour
    and its defs-key wording in the message.  The resolved entry is returned
    for callers that must also check it against the reference itself.
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
            return target
        current = next_key


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
        f"{node_desc} references function_id={fn_id!r} which is not in program.functions"
    )


# ---------------------------------------------------------------------------
# IrCatchHandler validation (deep tier)
# ---------------------------------------------------------------------------


def _validate_catch_handler(handler: IrCatchHandler, ctx: _Context) -> None:
    """Validate a catch handler: nominal/symbol cross-references (deep) + body."""
    if ctx.deep:
        if handler.nominal is not None:
            _check_nominal_in_table(handler.nominal, ctx)
            if ctx.program.nominals[handler.nominal].kind is not NominalKind.EXCEPTION:
                raise InvalidIrError(
                    f"IrCatchHandler references non-EXCEPTION nominal {handler.nominal!r}"
                )
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


def _is_payload_candidate(symbol: SymbolId, ctx: _Context) -> bool:
    """Report whether *symbol* has the shape an ``IrCase`` arm binding must have.

    A cheap over-approximation of the payload inventory, usable before the
    traversal has met every arm: other lowering temporaries share this shape, so
    :func:`_check_payload_dominance` re-tests the recorded requirements against
    the exact set of bound payload symbols.
    """
    descriptor = ctx.program.symbols[symbol]
    return descriptor.synthetic and not descriptor.mutable and descriptor.public_name is None


def _case_family(arm: IrCaseArm) -> tuple[str, object]:
    if isinstance(arm.key, IrNominalCaseKey):
        return "nominal", None
    return "literal", arm.key.kind


def _validate_case_arm(arm: IrCaseArm, ctx: _Context) -> None:
    match arm.key:
        case IrNominalCaseKey(nominal=nominal):
            if ctx.deep:
                _check_record_nominal(nominal, ctx, "IrNominalCaseKey")
                valid_fields = set(ctx.program.nominals[nominal].fields)
            else:
                valid_fields = None
        case IrLiteralCaseKey() as key:
            if not is_canonical_literal_scalar(key.kind, key.scalar_value):
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
            raise InvalidIrError(f"IrCaseArm binds unknown immediate field {field_name!r}")
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
            arm.key.scalar_value for arm in node.arms if isinstance(arm.key, IrLiteralCaseKey)
        }
        if bool_keys != {True, False}:
            raise InvalidIrError("IrCase has an incomplete boolean domain without a default")
    elif family is None or family[0] == "literal":
        raise InvalidIrError("IrCase over an open domain requires a default")


# ---------------------------------------------------------------------------
# Closed-union expression dispatcher
# ---------------------------------------------------------------------------


def _validate_expr(
    node: IrExpr, ctx: _Context, *, merge_requirements: bool = True
) -> frozenset[SymbolId]:
    """Validate a DAG node once and record its free payload requirements.

    Dominance itself is not decided here: whether a required symbol is a case
    payload is only known once every arm has been seen, so each visit narrows
    the node's dominator set and :func:`_check_payload_dominance` renders the
    verdict after the traversal.
    """
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
        ctx.payload_requirements[identifier] = requirements

    if ctx.check_payload_dominance:
        dominators = ctx.dominators.get(identifier)
        ctx.dominators[identifier] = (
            ctx.dominating_payload_symbols
            if dominators is None
            else dominators & ctx.dominating_payload_symbols
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

        case IrResource(path=path):
            _validate_location(node.location, ctx)
            if not Path(path).is_absolute():
                raise InvalidIrError("resource paths must be absolute")

        case IrConstUnit():
            _validate_location(node.location, ctx)

        case IrConstJsonNull():
            _validate_location(node.location, ctx)

        case IrMakeArray() | IrMakeJsonArray():
            _validate_location(node.location, ctx)
            for item in node.items:
                _validate_expr(item, ctx)

        case IrMakeDict() | IrMakeJsonObject():
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
                if ctx.check_payload_dominance and _is_payload_candidate(node.symbol, ctx):
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
            # DIV op requires DECIMAL kind (DIV always returns decimal)
            if op is ArithOp.DIV and kind is not ArithKind.DECIMAL:
                raise InvalidIrError(f"IrArith: DIV op requires DECIMAL kind, got kind={kind!r}")
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
                    f"IrCompare: ordering op {op!r} requires INT/DECIMAL/TEXT kind, got STRUCTURAL"
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
                raise InvalidIrError(f"IrUnary NOT: kind must be None, got kind={kind!r}")
            if op is UnaryOp.NEG and kind is None:
                raise InvalidIrError("IrUnary NEG: kind must not be None")
            _validate_expr(val, ctx)

        case IrField(value=val, nominal=nominal, field=field, mode=mode):
            _validate_location(node.location, ctx)
            if not field:
                raise InvalidIrError("IrField field must be non-empty")
            if ctx.deep:
                _check_nominal_in_table(nominal, ctx)
                _check_nominal_field(nominal, field, mode, ctx, "IrField")
            _validate_expr(val, ctx)

        case IrFieldSet(value=val, nominal=nominal, field=field, new=new):
            _validate_location(node.location, ctx)
            if not field:
                raise InvalidIrError("IrFieldSet field must be non-empty")
            if ctx.deep:
                _check_mutable_record_field(nominal, field, ctx)
            _validate_expr(val, ctx)
            _validate_expr(new, ctx)

        case IrUpdateRecord(value=val, updates=updates):
            _validate_location(node.location, ctx)
            _validate_expr(val, ctx)
            for _fname, fexpr in updates:
                _validate_expr(fexpr, ctx)

        case IrIndex(kind=_kind, value=val, index=idx):
            _validate_location(node.location, ctx)
            _validate_expr(val, ctx)
            _validate_expr(idx, ctx)

        case IrIndexSet(container=container, kind=_kind, index=idx, value=val):
            _validate_location(node.location, ctx)
            _validate_expr(container, ctx)
            _validate_expr(idx, ctx)
            _validate_expr(val, ctx)

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
            _validate_constructor_fields(nominal, fields, ctx, "IrMakeRecord")

        case IrMakeException(nominal=nominal, fields=fields):
            _validate_location(node.location, ctx)
            _validate_constructor_fields(nominal, fields, ctx, "IrMakeException")

        case IrMakeConstructor(nominal=nominal):
            _validate_location(node.location, ctx)
            if ctx.deep:
                _check_record_nominal(nominal, ctx, "IrMakeConstructor")

        case IrNominalCast(nominals=nominals, value=val):
            _validate_location(node.location, ctx)
            if ctx.deep:
                if not nominals:
                    raise InvalidIrError("IrNominalCast requires at least one nominal")
                _check_downcast_nominals(nominals, ctx, "IrNominalCast")
            _validate_expr(val, ctx)

        case IrNominalIs(nominal=nominal, value=val):
            _validate_location(node.location, ctx)
            if ctx.deep:
                _check_nominal_is_target(nominal, ctx, "IrNominalIs")
            _validate_expr(val, ctx)

        case IrConvert(value=val, recipe=recipe):
            _validate_location(node.location, ctx)
            _check_recipe_consistency(
                recipe.strategy,
                recipe.json_schema,
                recipe.decode,
                recipe.defs,
                recipe.encode,
                recipe.encode_definitions,
            )
            if ctx.deep and recipe.decode is not None:
                _check_decode_nominals(recipe.decode, recipe.defs, ctx)
            if ctx.deep and recipe.encode is not None:
                _check_encode_nominals(recipe.encode, recipe.encode_definitions, ctx)
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
                    if ctx.check_payload_dominance and _is_payload_candidate(cap.symbol, ctx):
                        ctx.requirement_collectors[-1].add(cap.symbol)

        case IrDirectCall(function_id=fn_id, arguments=arguments):
            _validate_location(node.location, ctx)
            params: tuple[IrFunctionParam, ...] | None = None
            if ctx.deep:
                params = _resolve_callable_params(fn_id, ctx, "IrDirectCall")
                if len(arguments) != len(params):
                    raise InvalidIrError(
                        f"IrDirectCall to function_id={fn_id!r} has {len(arguments)}"
                        f" arguments but the function has {len(params)} parameters"
                    )
            context = f"IrDirectCall to function_id={fn_id!r}"
            for index, arg in enumerate(arguments):
                if isinstance(arg, UseDefault):
                    _validate_use_default_slot(
                        context,
                        "parameter",
                        arg,
                        index,
                        ctx,
                        params[index].default is not None if params is not None else None,
                    )
                else:
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

        case IrCopyValue(value=val):
            _validate_location(node.location, ctx)
            _validate_expr(val, ctx)

        case IrAsk(
            agent=agent_expr, prompt=prompt_expr, contract_id=contract_id, sandbox=sandbox_expr
        ):
            _validate_location(node.location, ctx)
            _validate_expr(agent_expr, ctx)
            _validate_expr(prompt_expr, ctx)
            _validate_expr(sandbox_expr, ctx)
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

        case IrSessionOpen(
            agent=agent_expr, transport=transport_expr, name=name_expr, sandbox=sandbox_expr
        ):
            _validate_location(node.location, ctx)
            _validate_expr(agent_expr, ctx)
            if transport_expr is not None:
                _validate_expr(transport_expr, ctx)
            _validate_expr(name_expr, ctx)
            _validate_expr(sandbox_expr, ctx)

        case IrSessionDefault():
            _validate_location(node.location, ctx)

        case IrSessionAsk(session=session_expr, prompt=prompt_expr, contract_id=contract_id):
            _validate_location(node.location, ctx)
            _validate_expr(session_expr, ctx)
            _validate_expr(prompt_expr, ctx)
            if ctx.deep:
                if contract_id not in ctx.program.contracts:
                    raise InvalidIrError(
                        f"IrSessionAsk references contract_id={contract_id!r}"
                        " which is not in program.contracts"
                    )
                if node.max_attempts < 1:
                    raise InvalidIrError(
                        f"IrSessionAsk has max_attempts={node.max_attempts!r} (must be >= 1)"
                    )

        case IrSessionOp(session=session_expr, op=op, arg=arg_expr):
            _validate_location(node.location, ctx)
            _validate_expr(session_expr, ctx)
            if not isinstance(op, IrSessionOpKind):
                raise InvalidIrError(f"IrSessionOp has unknown operation {op!r}")
            if op is IrSessionOpKind.SET_NAME and arg_expr is None:
                raise InvalidIrError("IrSessionOp set-name requires an argument")
            if (
                op not in {IrSessionOpKind.COMPACT, IrSessionOpKind.SET_NAME}
                and arg_expr is not None
            ):
                raise InvalidIrError(f"IrSessionOp {op} does not accept an argument")
            if arg_expr is not None:
                _validate_expr(arg_expr, ctx)

        case IrAskRequest(
            agent=agent_expr, prompt=prompt_expr, contract_id=contract_id, sandbox=sandbox_expr
        ):
            _validate_location(node.location, ctx)
            _validate_expr(agent_expr, ctx)
            _validate_expr(prompt_expr, ctx)
            _validate_expr(sandbox_expr, ctx)
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

        case IrExec(
            command=command_expr,
            env=env_expr,
            cwd=cwd_expr,
            timeout=timeout_expr,
            contract_id=contract_id,
            sandbox=sandbox_expr,
        ):
            _validate_location(node.location, ctx)
            _validate_expr(command_expr, ctx)
            _validate_expr(env_expr, ctx)
            _validate_expr(cwd_expr, ctx)
            _validate_expr(timeout_expr, ctx)
            _validate_expr(sandbox_expr, ctx)
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

        case IrBuiltinLoad(key=key):
            _validate_location(node.location, ctx)
            _validate_builtin_key(key, ctx)

        case IrBuiltinStore(key=key, value=store_value):
            _validate_location(node.location, ctx)
            _validate_builtin_key(key, ctx)
            _validate_expr(store_value, ctx)

        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


# ---------------------------------------------------------------------------
# Deep tier — program-table checks
# ---------------------------------------------------------------------------


def _validate_program_tables(ctx: _Context) -> None:
    """Run deep cross-reference checks on the top-level program tables."""
    program = ctx.program

    # 1. entry_module
    if program.entry_module not in program.modules:
        raise InvalidIrError(f"entry_module={program.entry_module!r} is not in program.modules")

    # 1b. module key/id consistency
    for key, em in program.modules.items():
        if em.module_id != key:
            raise InvalidIrError(
                f"program.modules entry keyed by {key!r} has module_id={em.module_id!r} (mismatch)"
            )

    # 1c. builtin-var defaults reference available bindings. Their expression
    # structure is checked with the module initializers below.
    for builtin_key in program.builtin_setting_defaults:
        _validate_builtin_key(builtin_key, ctx)

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
        if nom_desc.kind is NominalKind.RECORD:
            unknown = nom_desc.mutable_fields - set(nom_desc.fields)
            if unknown:
                raise InvalidIrError(
                    f"record descriptor declares mutable fields {sorted(unknown)!r} it does"
                    f" not declare as fields for nominal {nom_key!r}"
                )
        elif nom_desc.mutable_fields:
            raise InvalidIrError(
                "enum and exception descriptors must have empty mutable_fields "
                f"for nominal {nom_key!r}"
            )

        if nom_desc.kind in (NominalKind.RECORD, NominalKind.EXCEPTION):
            if len(nom_desc.field_defaults) != len(nom_desc.fields):
                raise InvalidIrError(
                    f"{nom_desc.kind!r} descriptor for nominal {nom_key!r} has"
                    f" {len(nom_desc.field_defaults)} field_defaults but"
                    f" {len(nom_desc.fields)} fields"
                )
        elif nom_desc.field_defaults:
            raise InvalidIrError(
                f"enum descriptor for nominal {nom_key!r} must have empty field_defaults"
            )

        if nom_desc.kind is NominalKind.EXCEPTION:
            visited: set[NominalId] = {nom_key}
            current = nom_desc.base
            while current is not None:
                if current in visited:
                    raise InvalidIrError(
                        f"exception descriptor base chain starting at nominal {nom_key!r} is cyclic"
                    )
                visited.add(current)
                base_desc = program.nominals.get(current)
                if base_desc is None or base_desc.kind is not NominalKind.EXCEPTION:
                    raise InvalidIrError(
                        f"exception descriptor for nominal {nom_key!r} has base={current!r},"
                        " which is not an exception descriptor in program.nominals"
                    )
                current = base_desc.base
        elif nom_desc.base is not None:
            raise InvalidIrError(
                f"{nom_desc.kind!r} descriptor for nominal {nom_key!r} must not set base"
            )

        if nom_desc.kind is NominalKind.ENUM:
            variant_names: set[str] = set()
            member_nominals: set[NominalId] = set()
            for variant in nom_desc.variants:
                if variant.name in variant_names:
                    raise InvalidIrError(
                        f"enum descriptor has duplicate variant name {variant.name!r}"
                    )
                variant_names.add(variant.name)
                if variant.member in member_nominals:
                    raise InvalidIrError(
                        f"enum descriptor reuses member record nominal {variant.member!r}"
                    )
                member_nominals.add(variant.member)
                member = program.nominals.get(variant.member)
                if member is None or member.kind is not NominalKind.RECORD:
                    raise InvalidIrError(
                        f"enum variant {variant.name!r} links non-record member {variant.member!r}"
                    )
                if member.fields != variant.fields:
                    raise InvalidIrError(
                        f"enum variant {variant.name!r} fields disagree with member"
                        f" {variant.member!r}"
                    )

    # 4. functions table consistency
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
                _validate_expr(param.default, ctx)
        match fn_desc.impl:
            case IrFunctionBody(body=body):
                _validate_expr(body, ctx)
            case ExternFunctionBody():
                pass
            case other:  # pragma: no cover
                assert_never(other)

    # 5. program-entry maps — each source declaration maps to a registered
    #    symbol and its matching public function. Together the maps are a
    #    bidirectional index of linked ``program def`` declarations.
    program_entry_symbols: set[SymbolId] = set()
    for declaration_id, symbol in program.program_symbols.items():
        if symbol in program_entry_symbols:
            raise InvalidIrError(
                f"program_symbols entries map multiple declaration IDs to symbol_id={symbol!r}"
            )
        program_entry_symbols.add(symbol)
        if symbol not in program.symbols:
            raise InvalidIrError(
                f"program_symbols entry for declaration_id={declaration_id!r} references"
                f" symbol_id={symbol!r} which is not in program.symbols"
            )
        if symbol not in program.program_functions:
            raise InvalidIrError(
                f"program_symbols entry for declaration_id={declaration_id!r} references"
                f" symbol_id={symbol!r} which has no program_functions entry"
            )

    synthetic_main_functions = tuple(
        function for function in program.functions.values() if function.is_synthetic_main
    )
    if program.synthetic_main_symbol is None:
        if synthetic_main_functions:
            raise InvalidIrError("synthetic main function exists without synthetic_main_symbol")
    elif (
        len(synthetic_main_functions) != 1
        or synthetic_main_functions[0].function_symbol != program.synthetic_main_symbol
    ):
        raise InvalidIrError(
            "synthetic_main_symbol must identify the sole marked synthetic main function"
        )
    elif program.synthetic_main_symbol not in program_entry_symbols:
        raise InvalidIrError("synthetic_main_symbol must be a linked program entry")
    else:
        synthetic_function_id = program.program_functions.get(program.synthetic_main_symbol)
        synthetic_function = (
            program.functions.get(synthetic_function_id)
            if synthetic_function_id is not None
            else None
        )
        if synthetic_function is None or not synthetic_function.is_synthetic_main:
            raise InvalidIrError(
                "synthetic_main_symbol must resolve through program_functions "
                "to its marked synthetic main descriptor"
            )

    for symbol, function_id in program.program_functions.items():
        if symbol not in program.symbols:
            raise InvalidIrError(
                f"program_functions entry for symbol_id={symbol!r} is not in program.symbols"
            )
        function = program.functions.get(function_id)
        if function is None:
            raise InvalidIrError(
                f"program_functions entry for symbol_id={symbol!r} references"
                f" function_id={function_id!r} which is not in program.functions"
            )
        if function.function_symbol != symbol:
            raise InvalidIrError(
                f"program_functions entry for symbol_id={symbol!r} references"
                f" function_id={function_id!r} whose function_symbol="
                f"{function.function_symbol!r} differs"
            )
        if not isinstance(function.impl, IrFunctionBody):
            raise InvalidIrError(
                f"program_functions entry for symbol_id={symbol!r} references"
                f" function_id={function_id!r} without an IrFunctionBody implementation"
            )
        if symbol not in program_entry_symbols:
            raise InvalidIrError(
                f"program_functions entry for symbol_id={symbol!r} has no program_symbols entry"
            )

    # 11. program_signatures — keyed by exactly the program_functions symbols;
    #     each entry's parameter count and required-ness agree with its
    #     function descriptor's own declared IrFunctionParam list.
    for symbol in program.program_functions:
        if symbol not in program.program_signatures:
            raise InvalidIrError(
                f"program_functions entry for symbol_id={symbol!r} has no program_signatures entry"
            )
    for symbol, program_params in program.program_signatures.items():
        if symbol not in program.program_functions:
            raise InvalidIrError(
                f"program_signatures entry for symbol_id={symbol!r} is not"
                " a linked program_functions entry"
            )
        function = program.functions[program.program_functions[symbol]]
        if len(program_params) != len(function.params):
            raise InvalidIrError(
                f"program_signatures entry for symbol_id={symbol!r} declares"
                f" {len(program_params)} parameter(s), but its function descriptor"
                f" declares {len(function.params)}"
            )
        for program_param, function_param in zip(program_params, function.params, strict=True):
            if program_param.required != (function_param.default is None):
                raise InvalidIrError(
                    f"program_signatures entry for symbol_id={symbol!r} parameter"
                    f" {program_param.name!r} disagrees with its function"
                    " descriptor's default"
                )
            _validate_program_param(program_param, ctx)

    # 12. program_configs — sparse: only a program def carrying '@config' has
    # an entry, with no target repeated within one program's entries, and
    # each value expression itself validates. A '@param' target's binding
    # is not cross-checked against param_bindings here: a REPL's growing
    # entry module may declare that '@param' in an earlier entry, so
    # param_bindings -- built from only the modules and item lists the
    # current lowering call actually walks -- can lack it even though the
    # checker legitimately resolved it against the session's persistent
    # scope. A std/config-rooted target has no such cross-entry escape
    # (engine settings are declared once, in the standard library), so it
    # is cheaply checked against the closed engine-key catalog instead.
    for symbol, entries in program.program_configs.items():
        if symbol not in program.program_functions:
            raise InvalidIrError(
                f"program_configs entry for symbol_id={symbol!r} is not"
                " a linked program_functions entry"
            )
        seen_targets: set[StaticBindingKey] = set()
        for target, value in entries:
            if target in seen_targets:
                raise InvalidIrError(
                    f"program_configs entry for symbol_id={symbol!r} has duplicate target"
                    f" {target!r}"
                )
            seen_targets.add(target)
            if (
                target[0] == STD_CONFIG_ID
                and not target[1]
                and not is_engine_builtin_var_key(target)
            ):
                raise InvalidIrError(
                    f"program_configs entry for symbol_id={symbol!r} target {target!r} names"
                    " no known engine setting"
                )
            _validate_expr(value, ctx)

    # 6. contracts table — each ContractRequest must be consistent.
    for cid, contract_req in program.contracts.items():
        _validate_contract_request(cid, contract_req, ctx)

    # (Sources table has no key/id consistency invariant beyond being keyed by
    # SourceId; key consistency is structural to dict construction.)


def _validate_program_param(param: IrProgramParam, ctx: _Context) -> None:
    """Validate a single ``IrProgramParam`` descriptor.

    Called only from :func:`_validate_program_tables`, itself run only in the
    deep tier.
    """
    _check_decode_nominals(param.external_decoder.decode, param.external_decoder.defs, ctx)


def _validate_contract_request(
    cid: ContractId,
    req: ContractRequest,
    ctx: _Context,
) -> None:
    """Validate a ContractRequest entry (deep tier)."""
    has_decode_fields = req.json_schema is not None or req.decode is not None or bool(req.defs)
    if req.is_unit or req.codec_name == "text":
        if has_decode_fields:
            raise InvalidIrError(f"ContractRequest {cid!r} must not carry json_schema/decode/defs")
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
# Payload dominance verdict (deep tier)
# ---------------------------------------------------------------------------


def _check_payload_dominance(ctx: _Context) -> None:
    """Check that every visited expression is dominated by the payloads it needs.

    Run once the traversal has met every arm, so ``ctx.payload_symbols`` is
    complete and the recorded requirements can be reduced to actual payload
    symbols. A node's dominators are the intersection over all incoming paths,
    so requiring a payload it does not dominate means at least one path to the
    node fails to bind that payload — independent of traversal order, which
    preserves DAG sharing.
    """
    for identifier, requirements in ctx.payload_requirements.items():
        missing = (requirements & ctx.payload_symbols) - ctx.dominators[identifier]
        if missing:
            symbol = next(iter(missing))
            raise InvalidIrError(
                f"IrLoad references payload symbol_id={symbol.value!r} outside a binding IrCaseArm"
            )


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
    ctx = _Context(program, deep=deep, check_payload_dominance=deep)

    if deep:
        _validate_program_tables(ctx)
        for nominal, field_encodes in program.exception_field_encodes.items():
            descriptor = program.nominals.get(nominal)
            if descriptor is None or descriptor.kind is not NominalKind.EXCEPTION:
                raise InvalidIrError(
                    "exception field encodes reference an unregistered non-exception nominal "
                    f"{nominal!r}"
                )
            field_names: set[str] = set()
            json_names: set[str] = set()
            for field_encode in field_encodes:
                if field_encode.field_name in field_names:
                    raise InvalidIrError(
                        f"exception field encodes duplicate field {field_encode.field_name!r}"
                    )
                field_names.add(field_encode.field_name)
                if field_encode.field_name not in descriptor.fields:
                    raise InvalidIrError(
                        "exception field encodes reference unknown field "
                        f"{field_encode.field_name!r} of nominal {nominal!r}"
                    )
                if field_encode.json_name in json_names:
                    raise InvalidIrError(
                        "exception field encodes duplicate JSON name "
                        f"{field_encode.json_name!r} of nominal {nominal!r}"
                    )
                json_names.add(field_encode.json_name)
                if field_encode.plan is not None:
                    _check_encode_nominals(
                        field_encode.plan.root, field_encode.plan.definitions, ctx
                    )
            missing_fields = set(descriptor.fields) - field_names
            if missing_fields:
                raise InvalidIrError(
                    f"exception field encodes for nominal {nominal!r} omit fields "
                    f"{sorted(missing_fields)!r}"
                )

    for _module_id, em in program.modules.items():
        for node in em.initializers:
            _validate_expr(node, ctx)
    for default in program.builtin_setting_defaults.values():
        _validate_expr(default, ctx)

    if deep:
        _check_payload_dominance(ctx)
