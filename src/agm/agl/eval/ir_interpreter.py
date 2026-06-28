"""IR evaluator for the AgL typeless execution IR (M2-B).

``IrInterpreter`` executes an ``ExecutableProgram`` using the D5 per-invocation
frame / let-by-value / var-by-cell model.

Allowed imports (per M2 contract):
- ``agm.agl.ir.*``
- ``agm.agl.semantics.values`` (all value types, Cell, Frame)
- ``agm.agl.semantics.exceptions`` (AglRaise, make_builtin_exception)
- ``agm.agl.eval._decimal`` (shared pinned decimal context)
- ``agm.agl.runtime.serialize`` (value_to_json_obj for ToJson coercion)

NOT allowed: ``agm.agl.syntax``, ``agm.agl.scope``, ``agm.agl.typecheck``.
"""

from __future__ import annotations

import decimal
from collections.abc import Mapping
from typing import TYPE_CHECKING, assert_never

from agm.agl.eval._decimal import AGL_DECIMAL_CONTEXT
from agm.agl.eval.arith import (
    AglDivisionByZero,
    add,
    contains,
    div,
    logical_not,
    mul,
    negate,
    order,
    sub,
    value_eq,
)
from agm.agl.eval.conversions import AglCastConversion, run_recipe
from agm.agl.eval.effects import EffectHandlers
from agm.agl.eval.indexing import AglIndexOutOfRange, AglMissingKey, index_get, index_set
from agm.agl.eval.matching import make_match_error as _make_match_error
from agm.agl.ir.contracts import ConversionFailureMode
from agm.agl.ir.ids import ContractId, FunctionId, Location, SymbolId
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
    IrContinue,
    IrConvert,
    IrDirectCall,
    IrExec,
    IrExpr,
    IrField,
    IrIf,
    IrIndex,
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
from agm.agl.ir.operations import (
    ArithOp,
    CmpOp,
    Coercion,
    IndexKind,
    IntToDecimal,
    MapDictValues,
    MapEnumFields,
    MapList,
    MapRecordFields,
    ToJson,
    UnaryOp,
)
from agm.agl.ir.program import ExecutableProgram, FunctionDescriptor
from agm.agl.ir.validate import InvalidIrError
from agm.agl.modules.ids import ModuleId
from agm.agl.runtime.agents import AgentRegistry
from agm.agl.runtime.codec import ParseResult, _parse_contract_output
from agm.agl.runtime.convert import StrictJsonParseError, parse_json_strict
from agm.agl.runtime.render import render_value
from agm.agl.runtime.serialize import value_to_json_obj
from agm.agl.runtime.trace import TraceStore, noop_trace
from agm.agl.semantics.exceptions import AglRaise
from agm.agl.semantics.exceptions import make_builtin_exception as _make_exc_value
from agm.agl.semantics.values import (
    AgentValue,
    BoolValue,
    Cell,
    ConstructorValue,
    DecimalValue,
    DictValue,
    EnumValue,
    ExceptionValue,
    Frame,
    IntValue,
    IrClosureValue,
    IteratorValue,
    JsonValue,
    ListValue,
    RecordValue,
    TextValue,
    UnitValue,
    Value,
)

if TYPE_CHECKING:
    from agm.agl.runtime.contract import OutputContract

__all__ = ["IrInterpreter", "_apply_coercion", "_make_exc_value"]


# ---------------------------------------------------------------------------
# Internal loop-control signals (not AglRaise — bypass IrTry catch handlers)
# ---------------------------------------------------------------------------


class _BreakSignal(Exception):
    """Raised by ``IrBreak`` evaluation; caught only by the enclosing ``IrLoop``.

    Propagates through ``IrTry`` bodies unchanged because those catch only
    ``AglRaise``.  This ensures a ``break`` inside a ``try`` block exits the
    loop, not the ``try``.
    """


class _ContinueSignal(Exception):
    """Raised by ``IrContinue`` evaluation; caught only by the enclosing ``IrLoop``.

    Propagates through ``IrTry`` bodies unchanged.  The ``IrLoop`` evaluator
    catches this and executes ``continue`` on its Python ``while True`` loop to
    start the next iteration.
    """


# ---------------------------------------------------------------------------
# Coercion helper — module-level for test access
# ---------------------------------------------------------------------------


def _apply_coercion(value: Value, coercion: Coercion) -> Value:
    """Apply a resolved ``Coercion`` to *value* and return the result.

    Switches on the closed ``Coercion`` union (D3/D4) — no runtime type
    sniffing of *value*; the coercion op is pre-resolved by the lowerer.

    Raises ``InvalidIrError`` when the value tag does not match the coercion
    (cannot occur in well-lowered IR; defensive check only).
    """
    match coercion:
        case IntToDecimal():
            if not isinstance(value, IntValue):
                raise InvalidIrError(
                    f"IntToDecimal coercion requires IntValue, got {type(value).__name__}"
                )
            return DecimalValue(decimal.Decimal(value.value))

        case ToJson():
            if isinstance(value, JsonValue):
                # Idempotent: already JSON, return as-is.
                return value
            return JsonValue(value_to_json_obj(value))

        case MapList(item=child_op):
            if not isinstance(value, ListValue):
                raise InvalidIrError(
                    f"MapList coercion requires ListValue, got {type(value).__name__}"
                )
            return ListValue(tuple(_apply_coercion(elem, child_op) for elem in value.elements))

        case MapDictValues(value=child_op):
            if not isinstance(value, DictValue):
                raise InvalidIrError(
                    f"MapDictValues coercion requires DictValue, got {type(value).__name__}"
                )
            return DictValue(
                {k: _apply_coercion(v, child_op) for k, v in value.entries.items()}
            )

        case MapRecordFields(fields=field_coercions):
            if not isinstance(value, RecordValue):
                raise InvalidIrError(
                    f"MapRecordFields coercion requires RecordValue, got {type(value).__name__}"
                )
            new_fields = dict(value.fields)
            for field_name, child_op in field_coercions:
                new_fields[field_name] = _apply_coercion(new_fields[field_name], child_op)
            return RecordValue(
                nominal=value.nominal, display_name=value.display_name, fields=new_fields
            )

        case MapEnumFields(variants=variant_coercions):
            if not isinstance(value, EnumValue):
                raise InvalidIrError(
                    f"MapEnumFields coercion requires EnumValue, got {type(value).__name__}"
                )
            # Find the entry for the active variant; if not listed, pass through.
            for variant_name, field_coercions in variant_coercions:
                if variant_name == value.variant:
                    new_fields = dict(value.fields)
                    for field_name, child_op in field_coercions:
                        new_fields[field_name] = _apply_coercion(new_fields[field_name], child_op)
                    return EnumValue(
                        nominal=value.nominal,
                        display_name=value.display_name,
                        variant=value.variant,
                        fields=new_fields,
                    )
            # Variant not listed in coercion — return unchanged.
            return value

        case _ as unreachable:  # pragma: no cover
            assert_never(unreachable)


# ---------------------------------------------------------------------------
# IrInterpreter
# ---------------------------------------------------------------------------


class IrInterpreter:
    """Evaluates an ``ExecutableProgram`` using the D5 frame/cell model.

    In M2 there is exactly one frame: the entry-module frame.  No function
    calls, closures, or per-iteration frames are present yet (M4).

    ``run()`` executes the entry module's initializers in order and returns
    ``{public_name: Value}`` for every top-level binding that has a
    ``public_name`` and is owned by the entry module.
    """

    DEFAULT_MAX_CALL_DEPTH: int = 256

    def __init__(
        self,
        program: ExecutableProgram,
        *,
        trace: TraceStore | None = None,
        max_call_depth: int = DEFAULT_MAX_CALL_DEPTH,
        param_values: Mapping[SymbolId, Value] | None = None,
        registry: AgentRegistry | None = None,
        strict_json: bool = False,
        shell_exec_timeout: float | None = None,
        host_contracts: Mapping[ContractId, "OutputContract"] | None = None,
        base_frame: Frame | None = None,
    ) -> None:
        self._program = program
        self._frames: list[Frame] = [base_frame if base_frame is not None else {}]
        self.initializer_values: list[Value] = []
        self.module_initializer_values: dict[ModuleId, list[Value]] = {}
        self._call_depth: int = 0
        self._trace: TraceStore = trace if trace is not None else noop_trace()
        self._max_call_depth: int = max_call_depth
        self._param_values: Mapping[SymbolId, Value] = (
            param_values if param_values is not None else {}
        )
        self._registry: AgentRegistry = registry if registry is not None else AgentRegistry(
            named={}, default_agent=None
        )
        self._strict_json: bool = strict_json
        self._shell_exec_timeout: float | None = shell_exec_timeout
        self._host_contracts: Mapping[ContractId, OutputContract] = (
            host_contracts if host_contracts is not None else {}
        )
        self._effects = EffectHandlers(self)

    def _parse_host_output(
        self, raw: str, contract_id: ContractId, *, effective_strict: bool
    ) -> ParseResult:
        contract = self._program.contracts[contract_id]
        host_contract = self._host_contracts.get(contract_id)
        if host_contract is None or contract.codec_name in {"text", "json"}:
            return _parse_contract_output(raw, contract, effective_strict=effective_strict)
        schema = (
            host_contract.json_schema
            if isinstance(host_contract.json_schema, dict)
            else None
        )
        return host_contract.codec.parse(
            raw,
            host_contract.target_type,
            strict_json=effective_strict,
            schema=schema,
        )

    @property
    def _frame(self) -> Frame:
        """Return the current (top-of-stack) frame."""
        return self._frames[-1]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _eval_expecting_bool(self, expr: IrExpr, context: str) -> bool:
        """Evaluate ``expr`` and require its result to be a ``BoolValue``."""
        value = self._eval(expr)
        if not isinstance(value, BoolValue):
            raise InvalidIrError(
                f"{context} expected BoolValue, got {type(value).__name__}"
            )
        return value.value

    def _eval_render_bool_option(self, expr: IrExpr, option_name: str) -> bool:
        return self._eval_expecting_bool(expr, f"IrRenderValue: {option_name}")

    def _index_failure(self, err: AglIndexOutOfRange | AglMissingKey) -> AglRaise:
        """Convert an index/key sentinel into an ``AglRaise`` with the appropriate fields.

        Centralises the three identical sentinel-wrapping sites (IrIndex handler and both
        steps of the IrAssign-with-path handler) so the exception message and field shapes
        are defined exactly once.
        """
        match err:
            case AglIndexOutOfRange():
                return AglRaise(
                    _make_exc_value(
                        "IndexError",
                        f"List index {err.index} out of range for length {err.length}",
                        trace_id=self._trace.new_event_id(),
                        index=IntValue(err.index),
                        length=IntValue(err.length),
                    ),
                )
            case AglMissingKey():
                return AglRaise(
                    _make_exc_value(
                        "KeyError",
                        f"Dict key {err.key!r} is missing",
                        trace_id=self._trace.new_event_id(),
                        key=TextValue(err.key),
                    ),
                )
            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    def _on_cast_failure(
        self, failure_mode: ConversionFailureMode, exc: AglCastConversion
    ) -> BoolValue:
        """Handle a fallible-cast failure per the conversion failure mode.

        ``RAISE_CAST_ERROR`` (``as``) raises a ``CastError`` matching the legacy
        field shapes; ``RETURN_BOOL`` (``as?``) yields ``BoolValue(False)``.
        """
        match failure_mode:
            case ConversionFailureMode.RAISE_CAST_ERROR:
                raise AglRaise(
                    _make_exc_value(
                        "CastError",
                        exc.message,
                        trace_id=self._trace.new_event_id(),
                        source_type=TextValue(exc.source_label),
                        target_type=TextValue(exc.target_label),
                        raw=TextValue(exc.raw),
                    ),
                )
            case ConversionFailureMode.RETURN_BOOL:
                return BoolValue(False)
            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    def _get_closure_for(self, fn_id: FunctionId) -> IrClosureValue:
        """Look up the IrClosureValue for fn_id from the function's symbol in the base frame."""
        desc = self._program.functions[fn_id]
        slot = self._frames[0].get(desc.function_symbol)
        if slot is None:
            raise InvalidIrError(
                f"IrDirectCall: function_symbol for fn_id={fn_id!r} not in base frame"
            )
        val = slot.value if isinstance(slot, Cell) else slot
        if not isinstance(val, IrClosureValue):
            raise InvalidIrError(
                f"IrDirectCall: function_symbol slot is not IrClosureValue,"
                f" got {type(val).__name__}"
            )
        return val

    def _bind_and_invoke(
        self,
        desc: "FunctionDescriptor",
        closure_val: IrClosureValue,
        bound_values: list[Value],
    ) -> Value:
        """Build a call frame, push it, evaluate the function body, pop the frame, return result.

        Shared by ``_execute_direct_call`` and ``_execute_indirect_call``.

        Function parameters are immutable in AgL (they can never be the target of an
        assignment), so they are bound by value — never boxed in a Cell.
        """
        call_frame: Frame = dict(closure_val.captures)
        for param, val in zip(desc.params, bound_values, strict=True):
            call_frame[param.symbol] = val

        self._frames.append(call_frame)
        self._call_depth += 1
        try:
            result = self._eval(desc.body)
        finally:
            self._call_depth -= 1
            self._frames.pop()

        return result

    def _install_entry_function_closures(self) -> None:
        """Pre-install available entry-module function closures for param defaults."""
        entry_module = self._program.modules[self._program.entry_module]
        for node in entry_module.initializers:
            match node:
                case IrBind(
                    symbol=sym,
                    value=IrMakeClosure(function_id=fn_id, captures=()) as closure_node,
                ):
                    desc = self._program.functions.get(fn_id)
                    if desc is None or desc.function_symbol != sym:
                        continue
                    value = self._eval(closure_node)
                    self._frames[0][sym] = value
                case _:
                    continue

    def _check_call_depth(self) -> None:
        if self._call_depth >= self._max_call_depth:
            raise AglRaise(
                _make_exc_value(
                    "RecursionError",
                    f"Maximum call depth ({self._max_call_depth}) exceeded",
                    trace_id=self._trace.new_event_id(),
                    limit=IntValue(self._max_call_depth),
                )
            )

    def _execute_direct_call(
        self,
        fn_id: FunctionId,
        arguments: "tuple[IrExpr | UseDefault, ...]",
        location: Location,
    ) -> Value:
        """Execute a direct call to a named user function.

        Depth check → evaluate arguments (UseDefault uses a captures frame) →
        ``_bind_and_invoke``.
        """
        self._check_call_depth()
        desc = self._program.functions[fn_id]
        closure_val = self._get_closure_for(fn_id)

        bound_values: list[Value] = []
        for param, arg in zip(desc.params, arguments, strict=True):
            if isinstance(arg, UseDefault):
                assert param.default is not None, (
                    "UseDefault arg but param has no default (lowerer bug)"
                )
                self._frames.append(dict(closure_val.captures))
                try:
                    val = self._eval(param.default)
                finally:
                    self._frames.pop()
            else:
                val = self._eval(arg)
            bound_values.append(val)

        return self._bind_and_invoke(desc, closure_val, bound_values)

    def _execute_indirect_call(
        self,
        callee_expr: IrExpr,
        arguments: "tuple[IrExpr, ...]",
        location: Location,
    ) -> Value:
        """Execute an indirect (value) call.

        Evaluation order (mirrors ``_apply_closure``):
        1. Evaluate the callee in the current frame.
        2. Depth-limit check (AFTER callee eval, BEFORE arg binding).
        3. Evaluate each positional arg in the caller frame, NO coercion.
        4. Defensive: use ``desc.params[i].default`` for omitted trailing params
           (evaluated in a captures frame).
        5. ``_bind_and_invoke``.
        """
        callee_val = self._eval(callee_expr)
        if isinstance(callee_val, ConstructorValue):
            constructor_desc = self._program.nominals[callee_val.nominal]
            field_names = (
                constructor_desc.fields
                if callee_val.variant is None
                else next(
                    v.fields
                    for v in constructor_desc.variants
                    if v.name == callee_val.variant
                )
            )
            fields = {
                name: self._eval(argument)
                for name, argument in zip(field_names, arguments, strict=True)
            }
            if callee_val.variant is None:
                return RecordValue(
                    nominal=callee_val.nominal,
                    display_name=callee_val.display_name,
                    fields=fields,
                )
            return EnumValue(
                nominal=callee_val.nominal,
                display_name=callee_val.display_name,
                variant=callee_val.variant,
                fields=fields,
            )
        if not isinstance(callee_val, IrClosureValue):
            raise InvalidIrError(
                f"IrIndirectCall: callee evaluated to {type(callee_val).__name__},"
                " expected IrClosureValue"
            )
        desc = self._program.functions[callee_val.function_id]

        self._check_call_depth()

        # Evaluate each positional argument in the CALLER frame (no coercion).
        bound_values: list[Value] = []
        for i, param in enumerate(desc.params):
            if i < len(arguments):
                val = self._eval(arguments[i])
            elif param.default is not None:
                # Defensive: evaluate default in a captures frame.
                captures_frame: Frame = dict(callee_val.captures)
                self._frames.append(captures_frame)
                try:
                    val = self._eval(param.default)
                finally:
                    self._frames.pop()
            else:
                raise InvalidIrError(
                    f"IrIndirectCall: missing argument for parameter {i!r}"
                    " and no default available (lowerer bug)"
                )
            bound_values.append(val)

        return self._bind_and_invoke(desc, callee_val, bound_values)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Value]:
        """Execute all modules in order and return the entry module's public bindings.

        Installs entry-module params into the base frame BEFORE any module
        initializer runs, then iterates over all modules in insertion order
        (library modules first, entry last) executing each module's initializers.
        All evaluation runs under the pinned AgL decimal context.
        """
        with decimal.localcontext(AGL_DECIMAL_CONTEXT):
            self._install_entry_function_closures()
            # Install entry params into the base frame before any initializer.
            for ir_param in self._program.params:
                if ir_param.symbol in self._param_values:
                    # Host-provided value takes priority.
                    self._frames[0][ir_param.symbol] = self._param_values[ir_param.symbol]
                elif ir_param.default is not None:
                    # Evaluate the default expression in the base frame.
                    value = self._eval(ir_param.default)
                    self._frames[0][ir_param.symbol] = value
                else:
                    # Required param without a value — host-prep bug.
                    raise InvalidIrError(
                        f"Required param {ir_param.public_name!r} has no value;"
                        " the host must supply a value for required params before calling run()"
                    )

            for mod in self._program.modules.values():
                module_values = self.module_initializer_values.setdefault(mod.module_id, [])
                for node in mod.initializers:
                    try:
                        value = self._eval_initializer(node)
                        self.initializer_values.append(value)
                        module_values.append(value)
                    except AglRaise as exc:
                        if exc.span is None:
                            exc.span = node.location
                        raise
        return self._collect_results()

    def _eval_initializer(self, node: IrExpr) -> Value:
        match node:
            case IrBind(symbol=sym, value=IrMakeClosure(function_id=fn_id)):
                desc = self._program.functions.get(fn_id)
                if desc is not None and desc.function_symbol == sym:
                    slot = self._frames[0].get(sym)
                    if slot is not None:
                        return slot.value if isinstance(slot, Cell) else slot
            case _:
                pass
        return self._eval(node)

    # ------------------------------------------------------------------
    # Expression evaluator (closed IrExpr dispatch, D4)
    # ------------------------------------------------------------------

    def _eval(self, node: IrExpr) -> Value:
        """Evaluate *node* in the current frame and return its value.

        Dispatches over the closed ``IrExpr`` union with a structural ``match``
        whose final arm is ``assert_never`` so mypy exhaustiveness makes a
        missing case a compile-time error (D4).
        """
        match node:
            case IrConstInt(value=v):
                return IntValue(v)

            case IrConstDecimal(value=v):
                return DecimalValue(v)

            case IrConstBool(value=v):
                return BoolValue(v)

            case IrConstText(value=v):
                return TextValue(v)

            case IrConstUnit():
                return UnitValue()

            case IrConstJsonNull():
                return JsonValue(None)

            case IrMakeList(items=items):
                return ListValue(tuple(self._eval(item) for item in items))

            case IrMakeDict(entries=entries):
                result: dict[str, Value] = {}
                for key_expr, val_expr in entries:
                    key_val = self._eval(key_expr)
                    if not isinstance(key_val, TextValue):
                        raise InvalidIrError(
                            f"IrMakeDict key must evaluate to TextValue,"
                            f" got {type(key_val).__name__}"
                        )
                    result[key_val.value] = self._eval(val_expr)
                return DictValue(result)

            case IrLoad(symbol=sym):
                slot = self._frame.get(sym)
                if slot is None and self._frames[0] is not self._frame:
                    # Module-level bindings (let, var, function symbols) live in the base
                    # frame (frames[0]) and are always accessible — even from inside a
                    # function call frame that did not explicitly capture them.  This
                    # mirrors lexical scope-chain parent traversal.
                    slot = self._frames[0].get(sym)
                if slot is None:
                    raise InvalidIrError(
                        f"IrLoad: symbol_id={sym.value!r} is not bound in the frame"
                    )
                if isinstance(slot, Cell):
                    return slot.value
                return slot

            case IrBind(symbol=sym, value=val_expr):
                value = self._eval(val_expr)
                desc = self._program.symbols.get(sym)
                if desc is not None and desc.mutable:
                    self._frame[sym] = Cell(value)
                else:
                    self._frame[sym] = value
                return value

            case IrAssign(symbol=sym, path=path, value=val_expr):
                slot = self._frame.get(sym)
                if slot is None and self._frames[0] is not self._frame:
                    # Module vars live in the base frame and are intentionally
                    # not closure captures.
                    slot = self._frames[0].get(sym)
                if not isinstance(slot, Cell):
                    desc = self._program.symbols.get(sym)
                    if desc is None:
                        raise InvalidIrError(
                            f"IrAssign: symbol_id={sym.value!r} is not in program.symbols"
                        )
                    raise InvalidIrError(
                        f"IrAssign: symbol_id={sym.value!r}"
                        f" (public_name={desc.public_name!r}) is not a mutable var"
                    )
                if not path:
                    # Simple assignment.  An assignment statement yields unit
                    # The mutation is the
                    # side effect.
                    slot.value = self._eval(val_expr)
                    mutation_desc = self._program.symbols[sym]
                    self._trace.mutation(
                        name=mutation_desc.public_name or f"symbol#{sym.value}",
                        value=slot.value,
                        span=node.location,
                    )
                    return UnitValue()
                # Indexed assignment with non-empty path
                root = slot.value
                # Traverse all intermediate steps, collecting (container, index_val, kind)
                containers: list[tuple[Value, Value, IndexKind]] = []
                current = root
                for step in path[:-1]:
                    idx_val = self._eval(step.index)
                    try:
                        next_val = index_get(step.kind, current, idx_val)
                    except (AglIndexOutOfRange, AglMissingKey) as e:
                        raise self._index_failure(e)
                    containers.append((current, idx_val, step.kind))
                    current = next_val
                # Final step: evaluate index, validate via index_get
                final_step = path[-1]
                final_idx = self._eval(final_step.index)
                try:
                    index_get(final_step.kind, current, final_idx)
                except (AglIndexOutOfRange, AglMissingKey) as e:
                    raise self._index_failure(e)
                # Evaluate RHS once
                new_value = self._eval(val_expr)
                # Rebuild containers leaf-outward
                updated = index_set(final_step.kind, current, final_idx, new_value)
                for container, idx_val, kind in reversed(containers):
                    updated = index_set(kind, container, idx_val, updated)
                slot.value = updated
                mutation_desc = self._program.symbols[sym]
                self._trace.mutation(
                    name=mutation_desc.public_name or f"symbol#{sym.value}",
                    value=slot.value,
                    span=node.location,
                )
                # An assignment statement yields unit (parity with legacy).
                return UnitValue()

            case IrCoerce(value=val_expr, operation=op):
                value = self._eval(val_expr)
                return _apply_coercion(value, op)

            case IrSequence(items=items) | IrBlock(items=items):
                last: Value = UnitValue()
                for item in items:
                    last = self._eval(item)
                return last

            case IrArith(op=arith_op, kind=kind, lhs=lhs_expr, rhs=rhs_expr):
                lhs_val = self._eval(lhs_expr)
                rhs_val = self._eval(rhs_expr)
                try:
                    match arith_op:
                        case ArithOp.ADD:
                            return add(kind, lhs_val, rhs_val)
                        case ArithOp.SUB:
                            return sub(kind, lhs_val, rhs_val)
                        case ArithOp.MUL:
                            return mul(kind, lhs_val, rhs_val)
                        case ArithOp.DIV:
                            return div(lhs_val, rhs_val)
                        case _ as unreachable:  # pragma: no cover
                            assert_never(unreachable)
                except AglDivisionByZero:
                    raise AglRaise(
                        _make_exc_value(
                            "ArithmeticError",
                            "Division by zero",
                            trace_id=self._trace.new_event_id(),
                            operation=TextValue("/"),
                        )
                    )

            case IrCompare(op=cmp_op, kind=_kind, lhs=lhs_expr, rhs=rhs_expr):
                lhs_val = self._eval(lhs_expr)
                rhs_val = self._eval(rhs_expr)
                match cmp_op:
                    case CmpOp.EQ:
                        return BoolValue(value_eq(lhs_val, rhs_val))
                    case CmpOp.NEQ:
                        return BoolValue(not value_eq(lhs_val, rhs_val))
                    case CmpOp.LT | CmpOp.LE | CmpOp.GT | CmpOp.GE:
                        return BoolValue(order(cmp_op, lhs_val, rhs_val))
                    case _ as _unreachable_cmp:  # pragma: no cover
                        assert_never(_unreachable_cmp)

            case IrContains(kind=kind, item=item_expr, container=container_expr):
                item_val = self._eval(item_expr)
                container_val = self._eval(container_expr)
                return BoolValue(contains(kind, item_val, container_val))

            case IrAnd(lhs=lhs_expr, rhs=rhs_expr):
                lhs_val = self._eval(lhs_expr)
                if not isinstance(lhs_val, BoolValue):
                    raise InvalidIrError(
                        f"IrAnd: lhs is not BoolValue, got {type(lhs_val).__name__}"
                    )
                if not lhs_val.value:
                    return BoolValue(False)
                rhs_val = self._eval(rhs_expr)
                if not isinstance(rhs_val, BoolValue):
                    raise InvalidIrError(
                        f"IrAnd: rhs is not BoolValue, got {type(rhs_val).__name__}"
                    )
                return BoolValue(rhs_val.value)

            case IrOr(lhs=lhs_expr, rhs=rhs_expr):
                lhs_val = self._eval(lhs_expr)
                if not isinstance(lhs_val, BoolValue):
                    raise InvalidIrError(
                        f"IrOr: lhs is not BoolValue, got {type(lhs_val).__name__}"
                    )
                if lhs_val.value:
                    return BoolValue(True)
                rhs_val = self._eval(rhs_expr)
                if not isinstance(rhs_val, BoolValue):
                    raise InvalidIrError(
                        f"IrOr: rhs is not BoolValue, got {type(rhs_val).__name__}"
                    )
                return BoolValue(rhs_val.value)

            case IrUnary(op=unary_op, kind=kind, value=val_expr):
                val = self._eval(val_expr)
                match unary_op:
                    case UnaryOp.NOT:
                        if not isinstance(val, BoolValue):
                            raise InvalidIrError(
                                f"IrUnary NOT: expected BoolValue, got {type(val).__name__}"
                            )
                        return logical_not(val)
                    case UnaryOp.NEG:
                        if kind is None:
                            raise InvalidIrError("IrUnary NEG: kind must not be None")
                        if not isinstance(val, (IntValue, DecimalValue)):
                            raise InvalidIrError(
                                f"IrUnary NEG: expected numeric, got {type(val).__name__}"
                            )
                        return negate(kind, val)
                    case _ as _unreachable_unary:  # pragma: no cover
                        assert_never(_unreachable_unary)

            case IrField(value=val_expr, field=field_name):
                val = self._eval(val_expr)
                if not isinstance(val, (RecordValue, ExceptionValue)):
                    raise InvalidIrError(
                        f"IrField: expected RecordValue or ExceptionValue,"
                        f" got {type(val).__name__}"
                    )
                return val.fields[field_name]

            case IrIndex(kind=kind, value=val_expr, index=idx_expr):
                container = self._eval(val_expr)
                index_val = self._eval(idx_expr)
                try:
                    return index_get(kind, container, index_val)
                except (AglIndexOutOfRange, AglMissingKey) as e:
                    raise self._index_failure(e)

            case IrRenderTemplate(segments=segs):
                parts: list[str] = []
                for seg in segs:
                    match seg:
                        case IrTemplateText(text=t):
                            parts.append(t)
                        case IrTemplateValue(value=v_expr):
                            parts.append(render_value(self._eval(v_expr)))
                        case _ as unreachable_seg:  # pragma: no cover
                            assert_never(unreachable_seg)
                return TextValue("".join(parts))

            case IrMakeRecord(nominal=nominal, display_name=display_name, fields=fields):
                record_fields: dict[str, Value] = {
                    fname: self._eval(fexpr) for fname, fexpr in fields
                }
                return RecordValue(
                    nominal=nominal,
                    display_name=display_name,
                    fields=record_fields,
                )

            case IrMakeEnum(
                nominal=nominal, display_name=display_name, variant=variant, fields=fields
            ):
                enum_fields: dict[str, Value] = {
                    fname: self._eval(fexpr) for fname, fexpr in fields
                }
                return EnumValue(
                    nominal=nominal,
                    display_name=display_name,
                    variant=variant,
                    fields=enum_fields,
                )

            case IrMakeException(nominal=nominal, display_name=display_name, fields=fields):
                # Allocate ONE trace id per construction; reuse for all AutoTraceField slots.
                tid: TextValue = TextValue(self._trace.new_event_id())
                exc_fields: dict[str, Value] = {}
                for fname, field_slot in fields:
                    if isinstance(field_slot, AutoTraceField):
                        exc_fields[fname] = tid
                    else:
                        exc_fields[fname] = self._eval(field_slot)
                return ExceptionValue(
                    nominal=nominal,
                    display_name=display_name,
                    fields=exc_fields,
                )

            case IrMakeConstructor(nominal=nominal, display_name=display_name, variant=variant):
                return ConstructorValue(
                    nominal=nominal,
                    display_name=display_name,
                    variant=variant,
                )

            case IrVariantIs(variant=variant, value=val_expr, negated=negated):
                value = self._eval(val_expr)
                if not isinstance(value, EnumValue):
                    raise InvalidIrError(
                        f"IrVariantIs: value is not EnumValue, got {type(value).__name__}"
                    )
                return BoolValue((value.variant == variant) != negated)

            case IrConvert(value=val_expr, recipe=recipe, failure_mode=failure_mode):
                source_value = self._eval(val_expr)
                try:
                    converted = run_recipe(recipe, source_value)
                except AglCastConversion as exc:
                    return self._on_cast_failure(failure_mode, exc)
                if failure_mode is ConversionFailureMode.RETURN_BOOL:
                    return BoolValue(True)
                return converted

            case IrIf(branches=branches, has_else=has_else):
                for branch in branches:
                    if branch.cond is None:
                        # Else branch — always taken.
                        branch_val = self._eval(branch.body)
                        return branch_val if has_else else UnitValue()
                    cond_val = self._eval(branch.cond)
                    if not isinstance(cond_val, BoolValue):
                        raise InvalidIrError(
                            f"IrIf: branch condition evaluated to"
                            f" {type(cond_val).__name__}, expected BoolValue"
                        )
                    if cond_val.value:
                        branch_val = self._eval(branch.body)
                        return branch_val if has_else else UnitValue()
                # No branch matched and no else — yield unit.
                return UnitValue()

            case IrRaise(exc=exc_expr):
                exc_val = self._eval(exc_expr)
                if not isinstance(exc_val, ExceptionValue):
                    raise InvalidIrError(
                        f"IrRaise: exc evaluated to {type(exc_val).__name__},"
                        " expected ExceptionValue"
                    )
                raise AglRaise(exc_val, span=node.location)

            case IrTry(body=body_expr, handlers=handlers):
                try:
                    return self._eval(body_expr)
                except AglRaise as exc:
                    for handler in handlers:
                        if (
                            handler.display_name is None
                            or handler.display_name == exc.exc.display_name
                        ):
                            if handler.symbol is not None:
                                self._frame[handler.symbol] = exc.exc
                            return self._eval(handler.body)
                    raise

            case IrCase(subject=subject_expr, arms=arms):
                subject_val = self._eval(subject_expr)
                for arm in arms:
                    bindings = self._try_match(arm.plan, subject_val)
                    if bindings is not None:
                        self._frame.update(bindings)
                        return self._eval(arm.body)
                raise AglRaise(
                    _make_match_error(subject_val, trace_id=self._trace.new_event_id())
                )

            case IrLoop(body=body_expr):
                # Unconditional repeat — all loop logic (bound checks, until
                # guards, for/while clauses) is desugared into the body by the
                # lowerer.  The only exits are IrBreak (leave the loop)
                # and IrContinue (next iteration).  Both signals propagate through
                # IrTry bodies (which catch only AglRaise) to reach this handler.
                while True:
                    try:
                        self._eval(body_expr)
                    except _BreakSignal:
                        return UnitValue()
                    except _ContinueSignal:
                        continue

            case IrBreak():
                raise _BreakSignal()

            case IrContinue():
                raise _ContinueSignal()

            case IrIterInit(collection=collection_expr):
                coll = self._eval(collection_expr)
                if isinstance(coll, ListValue):
                    elements: list[Value] = list(coll.elements)
                elif isinstance(coll, DictValue):
                    elements = [TextValue(k) for k in coll.entries]
                elif isinstance(coll, TextValue):
                    elements = [TextValue(ch) for ch in coll.value]
                else:  # pragma: no cover
                    raise InvalidIrError(
                        f"IrIterInit: unexpected collection type {type(coll)!r}"
                    )
                return IteratorValue(elements=elements)

            case IrIterHasNext(iterator=iter_expr):
                it = self._eval(iter_expr)
                if not isinstance(it, IteratorValue):  # pragma: no cover
                    raise InvalidIrError(
                        f"IrIterHasNext: expected IteratorValue, got {type(it)!r}"
                    )
                return BoolValue(it.pos < len(it.elements))

            case IrIterNext(iterator=iter_expr):
                it = self._eval(iter_expr)
                if not isinstance(it, IteratorValue):  # pragma: no cover
                    raise InvalidIrError(
                        f"IrIterNext: expected IteratorValue, got {type(it)!r}"
                    )
                elem = it.elements[it.pos]
                it.pos += 1
                return elem

            case IrMakeClosure(function_id=fn_id, captures=captures):
                cap_slots: list[tuple[SymbolId, Value | Cell]] = []
                for cap in captures:
                    slot = self._frame.get(cap.symbol)
                    if slot is None:
                        raise InvalidIrError(
                            f"IrMakeClosure: capture symbol_id={cap.symbol.value!r}"
                            " not in frame"
                        )
                    if cap.by_cell:
                        if not isinstance(slot, Cell):
                            raise InvalidIrError(
                                f"IrMakeClosure: by_cell capture symbol_id={cap.symbol.value!r}"
                                " but slot is not Cell"
                            )
                        cap_slots.append((cap.symbol, slot))
                    else:
                        val = slot.value if isinstance(slot, Cell) else slot
                        cap_slots.append((cap.symbol, val))
                function_desc = self._program.functions[fn_id]
                return IrClosureValue(
                    function_id=fn_id,
                    captures=tuple(cap_slots),
                    param_labels=function_desc.param_labels,
                    arity=len(function_desc.params),
                    result_label=function_desc.result_label,
                )

            case IrDirectCall(function_id=fn_id, arguments=arguments):
                return self._execute_direct_call(fn_id, arguments, node.location)

            case IrIndirectCall(callee=callee_expr, arguments=arguments):
                return self._execute_indirect_call(callee_expr, arguments, node.location)

            case IrPrint(value=val_expr):
                val = self._eval(val_expr)
                rendered = render_value(val)
                print(rendered)
                self._trace.print_stmt(rendered=rendered, span=node.location)
                return UnitValue()

            case IrRenderValue(
                value=val_expr,
                pretty=pretty_expr,
                quote_strings=quote_strings_expr,
            ):
                pretty = (
                    self._eval_render_bool_option(pretty_expr, "pretty")
                    if pretty_expr is not None
                    else True
                )
                quote_strings = (
                    self._eval_render_bool_option(quote_strings_expr, "quote_strings")
                    if quote_strings_expr is not None
                    else True
                )
                return TextValue(
                    render_value(
                        self._eval(val_expr),
                        pretty=pretty,
                        quote_strings=quote_strings,
                    )
                )

            case IrParseJson(value=val_expr):
                val = self._eval(val_expr)
                if not isinstance(val, TextValue):
                    raise InvalidIrError(
                        f"IrParseJson: expected TextValue, got {type(val).__name__}"
                    )
                try:
                    obj = parse_json_strict(val.value)
                except StrictJsonParseError as exc:
                    raise AglRaise(
                        _make_exc_value(
                            "JsonParseError",
                            exc.message,
                            trace_id=self._trace.new_event_id(),
                            raw=TextValue(val.value),
                        ),
                        span=node.location,
                    ) from exc
                return JsonValue(obj)

            case IrAgentHandle(agent_name=agent_name):
                return AgentValue(name=agent_name)

            case IrAsk(
                agent=agent_expr,
                prompt=prompt_expr,
                contract_id=contract_id,
                max_attempts=max_attempts,
            ):
                try:
                    return self._effects.eval_ir_ask(
                        node, agent_expr, prompt_expr, contract_id, max_attempts
                    )
                except AglRaise as exc:
                    if exc.span is None:
                        exc.span = node.location
                    raise

            case IrAskRequest(agent=agent_expr, prompt=prompt_expr, contract_id=contract_id):
                return self._effects.eval_ir_ask_request(node, agent_expr, prompt_expr, contract_id)

            case IrExec(
                command=command_expr,
                contract_id=contract_id,
                max_attempts=max_attempts,
            ):
                try:
                    return self._effects.eval_ir_exec(node, command_expr, contract_id, max_attempts)
                except AglRaise as exc:
                    if exc.span is None:
                        exc.span = node.location
                    raise

            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    # ------------------------------------------------------------------
    # Pattern matching helper (M3f-B)
    # ------------------------------------------------------------------

    def _try_match(
        self, plan: IrMatchPlan, value: Value
    ) -> dict[SymbolId, Value] | None:
        """Try to match *value* against *plan*.

        Returns a dict of ``{SymbolId: Value}`` bindings on success, or
        ``None`` on mismatch.  Mirrors ``_match_pattern`` from the legacy
        interpreter — closed ``match``/``assert_never`` dispatch (D4).

        Defensive: ``IrVariantPlan`` and ``IrConstructorPlan`` raise
        ``InvalidIrError`` when applied to a non-``EnumValue`` (cannot occur
        in well-lowered IR).
        """
        match plan:
            case IrWildcardPlan():
                return {}

            case IrBindPlan(symbol=sym):
                return {sym: value}

            case IrLiteralPlan(value=val_expr):
                pat_val = self._eval(val_expr)
                return {} if value_eq(value, pat_val) else None

            case IrVariantPlan(variant=variant):
                if not isinstance(value, EnumValue):
                    raise InvalidIrError(
                        f"IrVariantPlan: value is not EnumValue,"
                        f" got {type(value).__name__}"
                    )
                return {} if value.variant == variant else None

            case IrConstructorPlan(variant=variant, fields=fields):
                if not isinstance(value, EnumValue):
                    raise InvalidIrError(
                        f"IrConstructorPlan: value is not EnumValue,"
                        f" got {type(value).__name__}"
                    )
                if value.variant != variant:
                    return None
                merged: dict[SymbolId, Value] = {}
                for fname, subplan in fields:
                    field_val = value.fields[fname]
                    sub_bindings = self._try_match(subplan, field_val)
                    if sub_bindings is None:
                        return None
                    merged.update(sub_bindings)
                return merged

            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    # ------------------------------------------------------------------
    # Result collection
    # ------------------------------------------------------------------

    def _collect_results(self) -> dict[str, Value]:
        """Return ``{public_name: Value}`` for symbols in the entry module frame.

        Only symbols that:
        1. Have a non-``None`` ``public_name`` in their ``SymbolDescriptor``.
        2. Are owned by the entry module.
        3. Are currently bound in the frame (``IrBind`` was executed for them).

        Cells are unwrapped; let-slots are returned directly.
        """
        entry_id: ModuleId = self._program.entry_module
        results: dict[str, Value] = {}
        for sym_id, desc in self._program.symbols.items():
            if desc.public_name is None:
                continue
            if desc.owner != entry_id:
                continue
            slot = self._frame.get(sym_id)
            if slot is None:
                continue
            if isinstance(slot, Cell):
                results[desc.public_name] = slot.value
            else:
                results[desc.public_name] = slot
        return results
