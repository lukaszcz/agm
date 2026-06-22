"""Single-module lowerer for the AgL typeless execution IR (M2-A).

Transforms a ``CheckedProgram`` into an ``ExecutableProgram`` for the M2 node
subset.  Every implicit coercion is inserted explicitly at compile time via
``compile_coercion``; the evaluator switches only on pre-resolved ``Coercion``
descriptors and never inspects value types at runtime.

M2 supported AST nodes
-----------------------
  Expressions
    UnitLit, IntLit, DecimalLit, BoolLit, NullLit, StringLit
    ListLit, DictLit
    VarRef
    Block

  Items (top-level and block-level)
    LetDecl, VarDecl, AssignStmt (simple name target only)
    Declarations that have no runtime action:
      RecordDef, EnumDef, TypeAlias, FuncDef, AgentDecl, ParamDecl,
      ProgramDecl, ConfigPragma, ImportDecl

Any AST node outside this set raises ``NotImplementedError`` with a clear
message.  A missing checker side-table entry is a compiler bug and raises
``AssertionError``.

Dispatch uses structural ``match`` with a final ``assert_never`` arm (D4).
"""

from __future__ import annotations

from typing import assert_never

from agm.agl.ir.ids import Location, NominalId, SourceId, SymbolId
from agm.agl.ir.nodes import (
    AutoTraceField,
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
    IrField,
    IrIndex,
    IrIndexStep,
    IrLoad,
    IrMakeConstructor,
    IrMakeDict,
    IrMakeEnum,
    IrMakeException,
    IrMakeList,
    IrMakeRecord,
    IrOr,
    IrRenderTemplate,
    IrTemplateText,
    IrTemplateValue,
    IrUnary,
)
from agm.agl.ir.operations import (
    ArithKind,
    ArithOp,
    CmpOp,
    CompareKind,
    ContainsKind,
    IndexKind,
    NumericKind,
    UnaryOp,
)
from agm.agl.ir.program import (
    ExecutableModule,
    ExecutableProgram,
    NominalDescriptor,
    NominalKind,
    SourceFile,
    SymbolDescriptor,
    VariantDescriptor,
)
from agm.agl.ir.validate import validate_ir
from agm.agl.lower.coercions import compile_coercion
from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID
from agm.agl.syntax.nodes import (
    AgentDecl,
    AssignStmt,
    AssignTarget,
    BinaryOp,
    BinOp,
    Block,
    BoolLit,
    Call,
    Case,
    Cast,
    ConfigPragma,
    DecimalLit,
    DictLit,
    Do,
    EnumDef,
    Expr,
    FieldAccess,
    FuncDef,
    If,
    ImportDecl,
    IndexAccess,
    InterpSegment,
    IntLit,
    IsTest,
    Item,
    Lambda,
    LetDecl,
    ListLit,
    NamedArg,
    NameTarget,
    NullLit,
    ParamDecl,
    ProgramDecl,
    Raise,
    RecordDef,
    StringLit,
    Template,
    TextSegment,
    Try,
    TypeAlias,
    UnaryNeg,
    UnaryNot,
    UnitLit,
    VarDecl,
    VarRef,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck.env import CheckedProgram
from agm.agl.typecheck.types import (
    BUILTIN_EXCEPTIONS,
    DecimalType,
    DictType,
    EnumType,
    ExceptionType,
    FunctionType,
    IntType,
    ListType,
    RecordType,
    TextType,
    Type,
)

__all__ = ["lower_program"]


# ---------------------------------------------------------------------------
# Internal lowerer state (one instance per lower_program call)
# ---------------------------------------------------------------------------


class _Lowerer:
    """Holds all mutable state for one lowering pass."""

    def __init__(
        self,
        checked: CheckedProgram,
        source_text: str,
        source_label: str,
    ) -> None:
        self._checked = checked
        self._source_text = source_text

        # Allocate the single source file entry.
        self._source_id = SourceId(0)
        self._source_file = SourceFile(
            display_name=source_label,
            normalized_text=source_text,
        )

        # Monotonic symbol counter; each declaration gets a fresh SymbolId.
        self._next_sym: int = 0

        # Maps decl_node_id → SymbolId so VarRef/AssignStmt can look up symbols.
        self._decl_to_sym: dict[int, SymbolId] = {}

        # Accumulated symbol descriptors.
        self._symbols: dict[SymbolId, SymbolDescriptor] = {}

        # Accumulated nominal descriptors (populated in lower()).
        self._nominals: dict[NominalId, NominalDescriptor] = {}

    # ------------------------------------------------------------------
    # SymbolId allocation
    # ------------------------------------------------------------------

    def _alloc_sym(self, decl_node_id: int, *, name: str, mutable: bool) -> SymbolId:
        """Allocate a fresh ``SymbolId`` for a declaration and register it."""
        sym = SymbolId(self._next_sym)
        self._next_sym += 1
        self._decl_to_sym[decl_node_id] = sym
        self._symbols[sym] = SymbolDescriptor(
            symbol_id=sym,
            mutable=mutable,
            public_name=name,
            owner=ENTRY_ID,
        )
        return sym

    def _sym_for_decl(self, decl_node_id: int) -> SymbolId:
        """Return the pre-allocated ``SymbolId`` for a declaration node."""
        sym = self._decl_to_sym.get(decl_node_id)
        assert sym is not None, (
            f"compiler bug: no SymbolId for decl_node_id={decl_node_id!r}; "
            "declaration must be visited before its references"
        )
        return sym

    # ------------------------------------------------------------------
    # Location helpers
    # ------------------------------------------------------------------

    def _loc(self, span: SourceSpan) -> Location:
        """Build an IR ``Location`` from an AST ``SourceSpan``."""
        return Location(
            source_id=self._source_id,
            start_offset=span.start_offset,
            end_offset=span.end_offset,
            start_line=span.start_line,
            start_col=span.start_col,
        )

    # ------------------------------------------------------------------
    # Binding-type helpers (mirror legacy _binding_type_for)
    # ------------------------------------------------------------------

    def _binding_type_for(self, decl_node_id: int) -> Type:
        """Return the checker-recorded type for a declaration node (compiler-error if missing)."""
        t = self._checked.type_env.get_binding_type(decl_node_id)
        assert t is not None, (
            f"compiler bug: no binding type for decl_node_id={decl_node_id!r}"
        )
        return t

    def _node_type(self, node_id: int) -> Type:
        """Return the checker-recorded type for an expression node (compiler-error if missing)."""
        t = self._checked.node_types.get(node_id)
        assert t is not None, (
            f"compiler bug: no node_type for node_id={node_id!r}"
        )
        return t

    # ------------------------------------------------------------------
    # Core lowering with optional coercion wrapping
    # ------------------------------------------------------------------

    def lower_coerced(self, node: Expr, expected: Type) -> IrExpr:
        """Lower *node* as an expression, then wrap in ``IrCoerce`` if needed.

        The node's own checked type is retrieved via ``node_types``; a coercion
        from that type to *expected* is compiled and, if non-``None``, wraps the
        result in ``IrCoerce``.
        """
        ir = self.lower_expr(node)
        own_type = self._node_type(node.node_id)
        op = compile_coercion(own_type, expected)
        if op is None:
            return ir
        return IrCoerce(location=self._loc(node.span), value=ir, operation=op)

    def lower_expr(self, node: Expr) -> IrExpr:
        """Lower an AST expression node to its own-typed IR (no outer coercion)."""
        match node:
            # ----------------------------------------------------------
            # Literals
            # ----------------------------------------------------------
            case IntLit(value=v, span=span):
                return IrConstInt(location=self._loc(span), value=v)

            case DecimalLit(value=v, span=span):
                return IrConstDecimal(location=self._loc(span), value=v)

            case BoolLit(value=v, span=span):
                return IrConstBool(location=self._loc(span), value=v)

            case StringLit(value=v, span=span):
                return IrConstText(location=self._loc(span), value=v)

            case NullLit(span=span):
                return IrConstJsonNull(location=self._loc(span))

            case UnitLit(span=span):
                return IrConstUnit(location=self._loc(span))

            # ----------------------------------------------------------
            # Container literals
            # ----------------------------------------------------------
            case ListLit() as list_node:
                return self._lower_list_lit(list_node)

            case DictLit() as dict_node:
                return self._lower_dict_lit(dict_node)

            # ----------------------------------------------------------
            # Variable reference — constructor ref or IrLoad
            # ----------------------------------------------------------
            case VarRef(node_id=nid, span=span):
                # Check for constructor reference FIRST (mirrors legacy _eval_var_ref).
                cref = self._checked.resolved.constructor_refs.get(nid)
                if cref is not None:
                    node_typ = self._node_type(nid)
                    if isinstance(node_typ, FunctionType):
                        # Constructor with fields used as a value → IrMakeConstructor.
                        nominal, display = self._nominal_for_cref_owner(cref.owner_name)
                        return IrMakeConstructor(
                            location=self._loc(span),
                            nominal=nominal,
                            display_name=display,
                            variant=cref.variant,
                        )
                    # Nullary constructor used as a value → construct immediately.
                    # AgL grammar requires ≥1 field in a record, so a nullary
                    # record VarRef is impossible under the current grammar.
                    return self._lower_nullary_constructor(  # pragma: no cover
                        nid, cref.owner_name, cref.variant, span
                    )

                # Cross-module constructor via BinderKind.constructor_binding.
                # Deferred to M5 (multi-module linking).
                from agm.agl.scope.symbols import BinderKind

                ref = self._checked.resolved.resolution.get(nid)
                # pragma: no cover — cross-module constructor binding (deferred to M5).
                if (  # pragma: no cover
                    ref is not None and ref.kind is BinderKind.constructor_binding
                ):
                    node_typ = self._node_type(nid)
                    if isinstance(node_typ, FunctionType):
                        nominal, display = self._nominal_for_resolution_name(ref.name)
                        return IrMakeConstructor(
                            location=self._loc(span),
                            nominal=nominal,
                            display_name=display,
                            variant=None,
                        )
                    return self._lower_nullary_constructor_by_name(nid, ref.name, None, span)

                assert ref is not None, (
                    f"compiler bug: no resolution for VarRef node_id={nid!r}"
                )
                sym = self._sym_for_decl(ref.decl_node_id)
                return IrLoad(location=self._loc(span), symbol=sym)

            # ----------------------------------------------------------
            # Block → IrBlock
            # ----------------------------------------------------------
            case Block(items=items, span=span):
                return self._lower_block(items, span)

            # ----------------------------------------------------------
            # Operator nodes — M3b
            # ----------------------------------------------------------
            case BinaryOp(op=op, left=left_expr, right=right_expr, span=span):
                return self._lower_binary_op(op, left_expr, right_expr, span)

            case UnaryNot(operand=operand_expr, span=span):
                return IrUnary(
                    location=self._loc(span),
                    op=UnaryOp.NOT,
                    kind=None,
                    value=self.lower_expr(operand_expr),
                )

            case UnaryNeg(operand=operand_expr, span=span):
                op_type = self._node_type(operand_expr.node_id)
                nkind = NumericKind.INT if isinstance(op_type, IntType) else NumericKind.DECIMAL
                return IrUnary(
                    location=self._loc(span),
                    op=UnaryOp.NEG,
                    kind=nkind,
                    value=self.lower_expr(operand_expr),
                )

            # ----------------------------------------------------------
            # Field access — qualified constructor ref or IrField (M3c/M3d)
            # ----------------------------------------------------------
            case FieldAccess(obj=obj_expr, field=field_name, span=span, node_id=nid):
                qcr = self._checked.resolved.qualified_constructor_refs.get(nid)
                if qcr is not None:
                    # qcr is (owner_name, member, owner_module_id | None)
                    owner_name, variant_name, _qcr_mid = qcr
                    node_typ = self._node_type(nid)
                    if isinstance(node_typ, FunctionType):
                        # With-fields variant used as value → IrMakeConstructor.
                        nominal, display = self._nominal_for_cref_owner(owner_name)
                        return IrMakeConstructor(
                            location=self._loc(span),
                            nominal=nominal,
                            display_name=display,
                            variant=variant_name,
                        )
                    # Nullary variant used as value → construct immediately.
                    return self._lower_nullary_constructor(nid, owner_name, variant_name, span)
                return IrField(
                    location=self._loc(span),
                    value=self.lower_expr(obj_expr),
                    field=field_name,
                )

            # ----------------------------------------------------------
            # Index access → IrIndex (M3c)
            # ----------------------------------------------------------
            case IndexAccess(obj=obj_expr, index=index_expr, span=span):
                container_type = self._node_type(obj_expr.node_id)
                kind = self._kind_for_container(container_type)
                return IrIndex(
                    location=self._loc(span),
                    kind=kind,
                    value=self.lower_expr(obj_expr),
                    index=self.lower_expr(index_expr),
                )

            # ----------------------------------------------------------
            # Template string → IrRenderTemplate (M3c)
            # ----------------------------------------------------------
            case Template(segments=segments, span=span):
                ir_segs: list[IrTemplateText | IrTemplateValue] = []
                for seg in segments:
                    if isinstance(seg, TextSegment):
                        ir_segs.append(IrTemplateText(text=seg.text))
                    elif isinstance(seg, InterpSegment):
                        ir_segs.append(IrTemplateValue(value=self.lower_expr(seg.expr)))
                    else:
                        assert_never(seg)  # pragma: no cover
                return IrRenderTemplate(location=self._loc(span), segments=tuple(ir_segs))

            # ----------------------------------------------------------
            # Call — constructor calls lowered here; all other calls deferred (M4)
            # ----------------------------------------------------------
            case Call(node_id=nid, span=span) as call_node:
                return self._lower_call(call_node, nid, span)

            # ----------------------------------------------------------
            # Unsupported M4+ nodes — deferred
            # ----------------------------------------------------------
            case (
                Lambda()
                | If()
                | Case()
                | Do()
                | Try()
                | Raise()
                | Cast()
                | IsTest()
            ):
                raise NotImplementedError(
                    f"Lowering of {type(node).__name__!r} is not yet implemented "
                    "(deferred to a later milestone)"
                )

            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    # ------------------------------------------------------------------
    # Container literal helpers
    # ------------------------------------------------------------------

    def _lower_list_lit(self, node: ListLit) -> IrMakeList:
        """Lower a ``ListLit``, applying element-level coercions."""
        own_type = self._node_type(node.node_id)
        assert isinstance(own_type, ListType), (
            f"compiler bug: ListLit has node_type {own_type!r}, expected ListType"
        )
        items = tuple(self.lower_coerced(e, own_type.elem) for e in node.elements)
        return IrMakeList(location=self._loc(node.span), items=items)

    def _lower_dict_lit(self, node: DictLit) -> IrMakeDict:
        """Lower a ``DictLit``, applying value-level coercions."""
        own_type = self._node_type(node.node_id)
        assert isinstance(own_type, DictType), (
            f"compiler bug: DictLit has node_type {own_type!r}, expected DictType"
        )
        ir_entries = tuple(
            (self.lower_expr(e.key), self.lower_coerced(e.value, own_type.value))
            for e in node.entries
        )
        return IrMakeDict(location=self._loc(node.span), entries=ir_entries)

    # ------------------------------------------------------------------
    # Operator helpers (M3b)
    # ------------------------------------------------------------------

    def _lower_binary_op(
        self, op: BinOp, left: Expr, right: Expr, span: SourceSpan
    ) -> IrExpr:
        """Lower a BinaryOp to the appropriate IR node."""
        loc = self._loc(span)

        if op is BinOp.AND:
            return IrAnd(location=loc, lhs=self.lower_expr(left), rhs=self.lower_expr(right))
        if op is BinOp.OR:
            return IrOr(location=loc, lhs=self.lower_expr(left), rhs=self.lower_expr(right))
        if op is BinOp.IN:
            return self._lower_in_op(left, right, loc)
        if op is BinOp.ADD or op is BinOp.SUB or op is BinOp.MUL:
            return self._lower_arith(op, left, right, loc)
        if op is BinOp.DIV:
            return self._lower_div(left, right, loc)
        if op is BinOp.EQ or op is BinOp.NEQ:
            return self._lower_equality(op, left, right, loc)
        if op is BinOp.LT or op is BinOp.LE or op is BinOp.GT or op is BinOp.GE:
            return self._lower_ordering(op, left, right, loc)
        assert_never(op)  # pragma: no cover

    def _lower_arith(
        self, op: BinOp, left: Expr, right: Expr, loc: Location
    ) -> IrArith:
        """Lower an arithmetic binary op (ADD/SUB/MUL)."""
        left_type = self._node_type(left.node_id)
        right_type = self._node_type(right.node_id)
        if isinstance(left_type, TextType) and isinstance(right_type, TextType):
            common: Type = TextType()
            kind = ArithKind.TEXT
        elif isinstance(left_type, DecimalType) or isinstance(right_type, DecimalType):
            common = DecimalType()
            kind = ArithKind.DECIMAL
        else:
            common = IntType()
            kind = ArithKind.INT
        arith_op_map: dict[BinOp, ArithOp] = {
            BinOp.ADD: ArithOp.ADD,
            BinOp.SUB: ArithOp.SUB,
            BinOp.MUL: ArithOp.MUL,
        }
        arith_op = arith_op_map[op]
        return IrArith(
            location=loc,
            op=arith_op,
            kind=kind,
            lhs=self.lower_coerced(left, common),
            rhs=self.lower_coerced(right, common),
        )

    def _lower_div(self, left: Expr, right: Expr, loc: Location) -> IrArith:
        """Lower a DIV op: always coerce both operands to decimal."""
        common: Type = DecimalType()
        return IrArith(
            location=loc,
            op=ArithOp.DIV,
            kind=ArithKind.DECIMAL,
            lhs=self.lower_coerced(left, common),
            rhs=self.lower_coerced(right, common),
        )

    def _lower_equality(
        self, op: BinOp, left: Expr, right: Expr, loc: Location
    ) -> IrCompare:
        """Lower EQ/NEQ: use STRUCTURAL kind with numeric widening if needed."""
        left_type = self._node_type(left.node_id)
        right_type = self._node_type(right.node_id)
        if isinstance(left_type, DecimalType) or isinstance(right_type, DecimalType):
            common: Type = DecimalType()
        else:
            common = left_type
        cmp_op = CmpOp.EQ if op is BinOp.EQ else CmpOp.NEQ
        return IrCompare(
            location=loc,
            op=cmp_op,
            kind=CompareKind.STRUCTURAL,
            lhs=self.lower_coerced(left, common),
            rhs=self.lower_coerced(right, common),
        )

    def _lower_ordering(
        self, op: BinOp, left: Expr, right: Expr, loc: Location
    ) -> IrCompare:
        """Lower LT/LE/GT/GE with kind based on operand types."""
        left_type = self._node_type(left.node_id)
        right_type = self._node_type(right.node_id)
        if isinstance(left_type, TextType) and isinstance(right_type, TextType):
            common: Type = TextType()
            kind = CompareKind.TEXT
        elif isinstance(left_type, DecimalType) or isinstance(right_type, DecimalType):
            common = DecimalType()
            kind = CompareKind.DECIMAL
        else:
            common = IntType()
            kind = CompareKind.INT
        cmp_op_map: dict[BinOp, CmpOp] = {
            BinOp.LT: CmpOp.LT,
            BinOp.LE: CmpOp.LE,
            BinOp.GT: CmpOp.GT,
            BinOp.GE: CmpOp.GE,
        }
        cmp_op = cmp_op_map[op]
        return IrCompare(
            location=loc,
            op=cmp_op,
            kind=kind,
            lhs=self.lower_coerced(left, common),
            rhs=self.lower_coerced(right, common),
        )

    def _lower_in_op(self, item: Expr, container: Expr, loc: Location) -> IrContains:
        """Lower the IN operator based on container type."""
        container_type = self._node_type(container.node_id)
        if isinstance(container_type, ListType):
            kind = ContainsKind.LIST
            item_ir = self.lower_coerced(item, container_type.elem)
        elif isinstance(container_type, DictType):
            kind = ContainsKind.DICT
            item_ir = self.lower_expr(item)
        elif isinstance(container_type, TextType):
            kind = ContainsKind.TEXT
            item_ir = self.lower_expr(item)
        else:  # pragma: no cover
            raise AssertionError(
                f"compiler bug: IN on non-container type {container_type!r}"
            )
        return IrContains(
            location=loc,
            kind=kind,
            item=item_ir,
            container=self.lower_expr(container),
        )

    # ------------------------------------------------------------------
    # Constructor lowering helpers (M3d)
    # ------------------------------------------------------------------

    def _nominal_for_cref_owner(self, owner_name: str) -> tuple[NominalId, str]:
        """Return (NominalId, display_name) for a constructor owner by name.

        For exceptions the nominal uses PRELUDE_ID; for records/enums it uses
        the type's own module_id (which equals ENTRY_ID for single-module programs).
        """
        typ = self._checked.type_env.get_type(owner_name)
        if isinstance(typ, RecordType):
            return NominalId(typ.module_id, typ.name), typ.name
        if isinstance(typ, EnumType):
            return NominalId(typ.module_id, typ.name), typ.name
        if isinstance(typ, ExceptionType):  # pragma: no cover
            # Exception constructors as first-class values are rejected by the checker.
            return NominalId(PRELUDE_ID, typ.name), typ.name
        # Fallback for generic types: get from GenericTypeDef template.  # pragma: no cover
        gdef = self._checked.type_env.get_generic_type(owner_name)  # pragma: no cover
        if gdef is not None:  # pragma: no cover
            tmpl = gdef.template  # pragma: no cover
            if isinstance(tmpl, RecordType):  # pragma: no cover
                return NominalId(tmpl.module_id, owner_name), owner_name  # pragma: no cover
            if isinstance(tmpl, EnumType):  # pragma: no cover
                return NominalId(tmpl.module_id, owner_name), owner_name  # pragma: no cover
        return NominalId(ENTRY_ID, owner_name), owner_name  # pragma: no cover

    def _nominal_for_resolution_name(self, name: str) -> tuple[NominalId, str]:  # pragma: no cover
        """Return (NominalId, display_name) for a constructor_binding resolution name.

        Called only from the cross-module constructor_binding VarRef path (M5).
        """
        return self._nominal_for_cref_owner(name)  # pragma: no cover

    def _lower_nullary_constructor(
        self,
        ref_node_id: int,
        owner_name: str,
        variant: str | None,
        span: "SourceSpan",
    ) -> IrExpr:
        """Lower a nullary constructor reference (value position) to an IrMake* node."""
        typ = self._checked.node_types.get(ref_node_id)
        return self._lower_constructor_from_type(typ, owner_name, variant, {}, span)

    def _lower_nullary_constructor_by_name(  # pragma: no cover
        self,
        ref_node_id: int,
        name: str,
        variant: str | None,
        span: "SourceSpan",
    ) -> IrExpr:
        """Lower a nullary constructor by name (for constructor_binding VarRef).

        Called only from the cross-module constructor_binding VarRef path (M5).
        """
        return self._lower_nullary_constructor(ref_node_id, name, variant, span)  # pragma: no cover

    def _lower_call(self, call_node: "Call", nid: int, span: "SourceSpan") -> IrExpr:
        """Lower a Call node.

        Constructor calls (VarRef or FieldAccess callee resolving to a constructor)
        are lowered to IrMakeRecord/IrMakeEnum/IrMakeException.  All other calls
        (functions, host, indirect) still raise NotImplementedError (deferred M4).
        """
        from agm.agl.scope.symbols import BinderKind

        callee = call_node.callee

        # (a) VarRef callee in constructor_refs
        if isinstance(callee, VarRef):
            cref = self._checked.resolved.constructor_refs.get(callee.node_id)
            if cref is not None:
                return self._lower_named_constructor_call(
                    nid, cref.owner_name, cref.variant, call_node.named_args, span
                )
            # (b) VarRef callee resolving via BinderKind.constructor_binding (M5).
            callee_ref = self._checked.resolved.resolution.get(callee.node_id)
            if (  # pragma: no cover
                callee_ref is not None
                and callee_ref.kind is BinderKind.constructor_binding
            ):
                return self._lower_named_constructor_call(  # pragma: no cover
                    nid, callee.name, None, call_node.named_args, span
                )

        # (c) FieldAccess callee in qualified_constructor_refs (enum variant call).
        # Any other callee type or non-constructor FieldAccess falls through to (d).
        elif isinstance(callee, FieldAccess):  # pragma: no branch
            qcr = self._checked.resolved.qualified_constructor_refs.get(callee.node_id)
            if qcr is not None:  # pragma: no branch — non-constructor FieldAccess call is M4+
                owner_name, variant_name, _qcr_mid = qcr
                return self._lower_named_constructor_call(
                    nid, owner_name, variant_name, call_node.named_args, span
                )

        # (d) Calling a ConstructorValue (first-class) — deferred to M4.
        # Non-constructor calls (functions/host/indirect) — deferred to M4.
        raise NotImplementedError(
            "Lowering of non-constructor Call is not yet implemented (deferred to M4)"
        )

    def _lower_named_constructor_call(
        self,
        result_node_id: int,
        owner_name: str,
        variant: str | None,
        named_args: "tuple[NamedArg, ...]",
        span: "SourceSpan",
    ) -> IrExpr:
        """Lower a constructor call with named arguments to an IrMake* node."""
        arg_exprs: dict[str, "Expr"] = {arg.name: arg.value for arg in named_args}
        typ = self._checked.node_types.get(result_node_id)
        return self._lower_constructor_from_type(typ, owner_name, variant, arg_exprs, span)

    def _lower_constructor_from_type(
        self,
        typ: "Type | None",
        owner_name: str,
        variant: str | None,
        arg_exprs: "dict[str, Expr]",
        span: "SourceSpan",
    ) -> IrExpr:
        """Build the IrMake* node for a constructor given its resolved checker type."""
        loc = self._loc(span)

        if isinstance(typ, RecordType):
            nominal = NominalId(typ.module_id, typ.name)
            # Build fields in declaration order from typ.fields.
            ir_fields: list[tuple[str, IrExpr]] = []
            for fname, ftype in typ.fields.items():
                ir_fields.append((fname, self.lower_coerced(arg_exprs[fname], ftype)))
            return IrMakeRecord(
                location=loc,
                nominal=nominal,
                display_name=typ.name,
                fields=tuple(ir_fields),
            )

        if isinstance(typ, ExceptionType):
            nominal = NominalId(PRELUDE_ID, typ.name)
            # ONE trace id allocation sentinel per construction (auto-fill any
            # declared field not present in arg_exprs).
            exc_fields: list[tuple[str, IrExpr | AutoTraceField]] = []
            for fname, ftype in typ.fields.items():
                if fname in arg_exprs:
                    exc_fields.append((fname, self.lower_coerced(arg_exprs[fname], ftype)))
                else:
                    exc_fields.append((fname, AutoTraceField()))
            return IrMakeException(
                location=loc,
                nominal=nominal,
                display_name=typ.name,
                fields=tuple(exc_fields),
            )

        if isinstance(typ, EnumType):
            assert variant is not None, "compiler bug: enum constructor must have variant"
            nominal = NominalId(typ.module_id, typ.name)
            variant_fields = typ.variants.get(variant, {})
            enum_fields: list[tuple[str, IrExpr]] = []
            for fname, ftype in variant_fields.items():
                # The checker enforces all variant fields are present in arg_exprs.
                enum_fields.append((fname, self.lower_coerced(arg_exprs[fname], ftype)))
            return IrMakeEnum(
                location=loc,
                nominal=nominal,
                display_name=typ.name,
                variant=variant,
                fields=tuple(enum_fields),
            )

        # Nullary constructor whose result type was not yet resolved (e.g. called
        # via a TypeVar-erased path) — look up owner type directly.  # pragma: no cover
        owner_typ = self._checked.type_env.get_type(owner_name)  # pragma: no cover
        if owner_typ is not None:  # pragma: no cover
            return self._lower_constructor_from_type(  # pragma: no cover
                owner_typ, owner_name, variant, arg_exprs, span
            )
        raise AssertionError(  # pragma: no cover
            f"compiler bug: cannot determine constructor type for {owner_name!r}"
        )

    # ------------------------------------------------------------------
    # Block helper
    # ------------------------------------------------------------------

    def _lower_block(
        self,
        items: tuple[Item, ...],
        span: SourceSpan,
    ) -> IrBlock:
        """Lower a ``Block``'s items to an ``IrBlock``."""
        ir_items = tuple(self.lower_item(it) for it in items)
        # Filter out None (items with no runtime action); validate non-empty.
        real: list[IrExpr] = [x for x in ir_items if x is not None]
        assert real, "compiler bug: lowered block has no runtime items"
        return IrBlock(location=self._loc(span), items=tuple(real))

    # ------------------------------------------------------------------
    # Item lowering
    # ------------------------------------------------------------------

    def lower_item(self, item: Item) -> IrExpr | None:
        """Lower a block item.

        Returns an ``IrExpr`` for nodes with runtime action, or ``None`` for
        purely compile-time declarations (type definitions, function defs, etc.)
        that have no IR representation in M2.

        The IrBlock construction must then filter out the ``None`` values.
        """
        match item:
            # ----------------------------------------------------------
            # Binders
            # ----------------------------------------------------------
            case LetDecl(name=name, value=rhs, span=span, node_id=nid):
                sym = self._alloc_sym(nid, name=name, mutable=False)
                binding_type = self._binding_type_for(nid)
                ir_val = self.lower_coerced(rhs, binding_type)
                return IrBind(location=self._loc(span), symbol=sym, value=ir_val)

            case VarDecl(name=name, value=rhs, span=span, node_id=nid):
                sym = self._alloc_sym(nid, name=name, mutable=True)
                binding_type = self._binding_type_for(nid)
                ir_val = self.lower_coerced(rhs, binding_type)
                return IrBind(location=self._loc(span), symbol=sym, value=ir_val)

            case AssignStmt(target=target, value=rhs, span=span, node_id=nid):
                return self._lower_assign(target, rhs, span, nid)

            # ----------------------------------------------------------
            # Declarations with no runtime action in M2
            # ----------------------------------------------------------
            case (
                FuncDef()
                | AgentDecl()
                | RecordDef()
                | EnumDef()
                | TypeAlias()
                | ParamDecl()
                | ProgramDecl()
                | ConfigPragma()
                | ImportDecl()
            ):
                return None

            # ----------------------------------------------------------
            # Expression items — lower as expr
            # ----------------------------------------------------------
            case _:
                # Anything else must be an expression.
                return self.lower_expr(item)

    def _lower_assign(
        self,
        target: AssignTarget,
        rhs: Expr,
        span: SourceSpan,
        assign_node_id: int,
    ) -> IrAssign:
        """Lower an assignment statement (simple name or indexed path)."""
        ref = self._checked.resolved.resolution.get(assign_node_id)
        assert ref is not None, (
            f"compiler bug: no resolution for AssignStmt node_id={assign_node_id!r}"
        )
        sym = self._sym_for_decl(ref.decl_node_id)

        if isinstance(target, NameTarget):
            slot_type = self._binding_type_for(ref.decl_node_id)
            ir_val = self.lower_coerced(rhs, slot_type)
            return IrAssign(
                location=self._loc(span),
                symbol=sym,
                path=(),
                value=ir_val,
            )

        # IndexTarget: flatten the index path into IrIndexStep list.
        root_type = self._binding_type_for(ref.decl_node_id)
        steps: list[IrIndexStep] = []
        container_type = self._collect_index_steps_from_obj(target.obj, root_type, steps)
        kind = self._kind_for_container(container_type)
        steps.append(IrIndexStep(
            kind=kind,
            index=self.lower_expr(target.index),
            location=self._loc(target.span),
        ))
        slot_type = self._elem_type_for_container(container_type)
        ir_val = self.lower_coerced(rhs, slot_type)
        return IrAssign(
            location=self._loc(span),
            symbol=sym,
            path=tuple(steps),
            value=ir_val,
        )

    def _kind_for_container(self, t: Type) -> IndexKind:
        """Return IndexKind for a container type (LIST or DICT)."""
        if isinstance(t, ListType):
            return IndexKind.LIST
        if isinstance(t, DictType):
            return IndexKind.DICT
        raise AssertionError(f"compiler bug: non-container type in index path: {t!r}")

    def _elem_type_for_container(self, t: Type) -> Type:
        """Return the element/value type for a container type."""
        if isinstance(t, ListType):
            return t.elem
        if isinstance(t, DictType):
            return t.value
        raise AssertionError(f"compiler bug: non-container type in index path: {t!r}")

    def _collect_index_steps_from_obj(
        self, obj: Expr, root_type: Type, out: list[IrIndexStep]
    ) -> Type:
        """Recursively descend into obj (VarRef or IndexAccess), collecting IrIndexSteps.

        Returns the container type at the deepest level (i.e., the type of ``obj``).
        """
        if isinstance(obj, VarRef):
            return root_type
        if isinstance(obj, IndexAccess):
            parent_type = self._collect_index_steps_from_obj(obj.obj, root_type, out)
            kind = self._kind_for_container(parent_type)
            out.append(IrIndexStep(
                kind=kind,
                index=self.lower_expr(obj.index),
                location=self._loc(obj.span),
            ))
            return self._elem_type_for_container(parent_type)
        raise AssertionError(  # pragma: no cover
            f"compiler bug: unexpected expr in indexed assignment path: {type(obj).__name__}"
        )

    # ------------------------------------------------------------------
    # Top-level entry point
    # ------------------------------------------------------------------

    def _build_nominals(self) -> None:
        """Populate ``self._nominals`` with all user-declared and built-in nominals.

        Adds:
        - All user-declared record/enum nominals from the entry module's type env.
          (Cross-module nominal aggregation is deferred to M5.)
        - All built-in exception descriptors keyed by NominalId(PRELUDE_ID, name).

        Out of scope: prelude record/enum descriptors (ExecResult, ParsePolicy, …)
        — those are constructed only via host ops (M6) and are not added here.
        """
        # User-declared nominals (entry module only — M5 handles cross-module).
        for name, typ in self._checked.type_env.non_builtin_type_items():
            if isinstance(typ, RecordType):
                nominal = NominalId(typ.module_id, name)
                self._nominals[nominal] = NominalDescriptor(
                    nominal=nominal,
                    display_name=name,
                    kind=NominalKind.RECORD,
                    fields=tuple(typ.fields.keys()),
                    variants=(),
                )
            elif isinstance(typ, EnumType):
                nominal = NominalId(typ.module_id, name)
                variants = tuple(
                    VariantDescriptor(name=vname, fields=tuple(vfields.keys()))
                    for vname, vfields in typ.variants.items()
                )
                self._nominals[nominal] = NominalDescriptor(
                    nominal=nominal,
                    display_name=name,
                    kind=NominalKind.ENUM,
                    fields=(),
                    variants=variants,
                )
            elif isinstance(typ, ExceptionType):  # pragma: no cover
                # User-declared exceptions are not supported by the current grammar.
                # Reserved for a future grammar extension.
                nominal = NominalId(PRELUDE_ID, name)  # pragma: no cover
                self._nominals[nominal] = NominalDescriptor(  # pragma: no cover
                    nominal=nominal,
                    display_name=name,
                    kind=NominalKind.EXCEPTION,
                    fields=tuple(typ.fields.keys()),
                    variants=(),
                )

        # Built-in exceptions: always keyed by NominalId(PRELUDE_ID, name).
        for exc_name, exc_type in BUILTIN_EXCEPTIONS.items():
            nominal = NominalId(PRELUDE_ID, exc_name)
            if nominal not in self._nominals:  # pragma: no cover — always true for builtins
                self._nominals[nominal] = NominalDescriptor(
                    nominal=nominal,
                    display_name=exc_name,
                    kind=NominalKind.EXCEPTION,
                    fields=tuple(exc_type.fields.keys()),
                    variants=(),
                )

    def lower(self) -> ExecutableProgram:
        """Lower the checked program to an ``ExecutableProgram``."""
        # Populate nominals before lowering items so that validate can check them.
        self._build_nominals()

        body = self._checked.resolved.program.body
        ir_items: list[IrExpr] = []
        for item in body.items:
            ir = self.lower_item(item)
            if ir is not None:
                ir_items.append(ir)

        entry_mod = ExecutableModule(
            module_id=ENTRY_ID,
            initializers=tuple(ir_items),
        )

        return ExecutableProgram(
            entry_module=ENTRY_ID,
            modules={ENTRY_ID: entry_mod},
            symbols=dict(self._symbols),
            nominals=dict(self._nominals),
            sources={self._source_id: self._source_file},
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def lower_program(
    checked: CheckedProgram,
    *,
    source_text: str,
    source_label: str,
    validate: bool = False,
) -> ExecutableProgram:
    """Lower a single-module ``CheckedProgram`` to an ``ExecutableProgram``.

    :param checked: the type-checked program to lower.
    :param source_text: the normalised source text (used in the sources table).
    :param source_label: human-readable label for the source (display_name).
    :param validate: when ``True``, run ``validate_ir`` before returning.
    :returns: the linked ``ExecutableProgram`` ready for evaluation.
    :raises NotImplementedError: for AST nodes outside the M2 subset.
    :raises AssertionError: for missing checker side-table entries (compiler bugs).
    """
    lowerer = _Lowerer(checked, source_text, source_label)
    program = lowerer.lower()
    if validate:
        validate_ir(program)
    return program
