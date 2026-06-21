"""IR evaluator for the AgL typeless execution IR (M2-B).

``IrInterpreter`` executes an ``ExecutableProgram`` using the D5 per-invocation
frame / let-by-value / var-by-cell model.  It is the seed of the evaluator that
ultimately replaces the legacy AST interpreter.

Allowed imports (per M2 contract):
- ``agm.agl.ir.*``
- ``agm.agl.values``
- ``agm.agl.eval.values`` (container/nominal tags)
- ``agm.agl.eval.frames`` (Cell, Frame)
- ``agm.agl.eval._decimal`` (shared pinned decimal context)
- ``agm.agl.runtime.serialize`` (value_to_json_obj for ToJson coercion)

NOT allowed: ``agm.agl.syntax``, ``agm.agl.scope``, ``agm.agl.typecheck``.
"""

from __future__ import annotations

import decimal
from typing import assert_never

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
from agm.agl.eval.exceptions import AglRaise
from agm.agl.eval.exceptions import make_builtin_exception as _make_exc_value
from agm.agl.eval.frames import Cell, Frame
from agm.agl.eval.values import (
    BoolValue,
    DecimalValue,
    DictValue,
    EnumValue,
    IntValue,
    JsonValue,
    ListValue,
    RecordValue,
    TextValue,
    UnitValue,
    Value,
)
from agm.agl.ir.nodes import (
    IrAnd,
    IrArith,
    IrAssign,
    IrBind,
    IrBlock,
    IrCoerce,
    IrCompare,
    IrConstBool,
    IrConstDecimal,
    IrConstInt,
    IrConstJsonNull,
    IrConstText,
    IrConstUnit,
    IrContains,
    IrExpr,
    IrLoad,
    IrMakeDict,
    IrMakeList,
    IrOr,
    IrSequence,
    IrUnary,
)
from agm.agl.ir.operations import (
    ArithOp,
    CmpOp,
    Coercion,
    IntToDecimal,
    MapDictValues,
    MapEnumFields,
    MapList,
    MapRecordFields,
    ToJson,
    UnaryOp,
)
from agm.agl.ir.program import ExecutableProgram
from agm.agl.ir.validate import InvalidIrError
from agm.agl.modules.ids import ModuleId
from agm.agl.runtime.serialize import value_to_json_obj
from agm.agl.runtime.trace import TraceStore, noop_trace

__all__ = ["IrInterpreter", "_apply_coercion", "_make_exc_value"]


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

    def __init__(self, program: ExecutableProgram, *, trace: TraceStore | None = None) -> None:
        self._program = program
        self._frame: Frame = {}
        self._trace: TraceStore = trace if trace is not None else noop_trace()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Value]:
        """Execute the entry module and return its public bindings.

        All evaluation runs under the pinned AgL decimal context so results
        are independent of the host's ambient ``decimal`` context (F7).
        """
        with decimal.localcontext(AGL_DECIMAL_CONTEXT):
            entry_mod = self._program.modules[self._program.entry_module]
            for node in entry_mod.initializers:
                self._eval(node)
        return self._collect_results()

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
                if path:
                    raise InvalidIrError(
                        "IrAssign with non-empty path is not supported in M2"
                    )
                slot = self._frame.get(sym)
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
                new_value = self._eval(val_expr)
                slot.value = new_value
                return new_value

            case IrCoerce(value=val_expr, operation=op):
                value = self._eval(val_expr)
                return _apply_coercion(value, op)

            case IrSequence(items=items):
                last: Value = UnitValue()
                for item in items:
                    last = self._eval(item)
                return last

            case IrBlock(items=items):
                last = UnitValue()
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
