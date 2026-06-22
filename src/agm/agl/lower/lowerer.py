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

from agm.agl._text import normalize_newlines
from agm.agl.ir.contracts import ConversionFailureMode
from agm.agl.ir.ids import FunctionId, Location, NominalId, SourceId, SymbolId
from agm.agl.ir.nodes import (
    AutoTraceField,
    IrAnd,
    IrArith,
    IrAssign,
    IrBind,
    IrBindPlan,
    IrBlock,
    IrCapture,
    IrCase,
    IrCaseArm,
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
    IrFunctionParam,
    IrIf,
    IrIfBranch,
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
    FunctionDescriptor,
    NominalDescriptor,
    NominalKind,
    SourceFile,
    SymbolDescriptor,
    VariantDescriptor,
)
from agm.agl.ir.validate import validate_ir
from agm.agl.lower.coercions import compile_coercion
from agm.agl.lower.conversions import compile_recipe
from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID, ModuleId
from agm.agl.scope.symbols import BinderKind, BindingRef
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
    CaseBranch,
    Cast,
    CatchClause,
    ConfigPragma,
    ConstructorPattern,
    DecimalLit,
    DictLit,
    Do,
    ElseSentinel,
    EnumDef,
    Expr,
    FieldAccess,
    FuncDef,
    If,
    IfBranch,
    ImportDecl,
    IndexAccess,
    IndexTarget,
    InterpSegment,
    IntLit,
    IsTest,
    Item,
    Lambda,
    LetDecl,
    ListLit,
    LiteralPattern,
    NamedArg,
    NameTarget,
    NullLit,
    ParamDecl,
    Pattern,
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
    VarPattern,
    VarRef,
    WildcardPattern,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.typecheck.env import CheckedProgram
from agm.agl.typecheck.types import (
    BUILTIN_EXCEPTIONS,
    CastKind,
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
        # Spans/offsets from the lexer are relative to the newline-normalized
        # source (scanner normalizes with ``normalize_newlines``), and the
        # legacy interpreter slices normalized text.  Normalize here so source
        # slices, the stored SourceFile, and every Location offset are all
        # consistent against the same string (idempotent if already normalized).
        self._source_text = normalize_newlines(source_text)

        # Allocate the single source file entry.
        self._source_id = SourceId(0)
        self._source_file = SourceFile(
            display_name=source_label,
            normalized_text=self._source_text,
        )

        # Monotonic symbol counter; each declaration gets a fresh SymbolId.
        self._next_sym: int = 0

        # Maps decl_node_id → SymbolId so VarRef/AssignStmt can look up symbols.
        self._decl_to_sym: dict[int, SymbolId] = {}

        # Accumulated symbol descriptors.
        self._symbols: dict[SymbolId, SymbolDescriptor] = {}

        # Accumulated nominal descriptors (populated in lower()).
        self._nominals: dict[NominalId, NominalDescriptor] = {}

        # Function allocation counter
        self._next_fn: int = 0
        # Accumulated FunctionDescriptors
        self._functions: dict[FunctionId, FunctionDescriptor] = {}
        # Maps FuncDef.node_id -> SymbolId (pre-allocated in phase 1)
        self._fn_node_to_sym: dict[int, SymbolId] = {}
        # Maps FuncDef.node_id -> FunctionId (pre-allocated in phase 1)
        self._fn_node_to_id: dict[int, FunctionId] = {}

    # ------------------------------------------------------------------
    # SymbolId allocation
    # ------------------------------------------------------------------

    def _alloc_sym(
        self,
        decl_node_id: int,
        *,
        name: str,
        mutable: bool,
        public: bool = True,
        owner: "ModuleId | FunctionId | None" = None,
    ) -> SymbolId:
        """Allocate a fresh ``SymbolId`` for a declaration and register it.

        When ``public`` is ``False`` the ``SymbolDescriptor.public_name`` is set
        to ``None`` so the symbol is not exposed in ``_collect_results``; this is
        used for catch-clause binders that live in the flat module frame but are
        not top-level exported bindings.
        """
        sym = SymbolId(self._next_sym)
        self._next_sym += 1
        self._decl_to_sym[decl_node_id] = sym
        self._symbols[sym] = SymbolDescriptor(
            symbol_id=sym,
            mutable=mutable,
            public_name=name if public else None,
            owner=owner if owner is not None else ENTRY_ID,
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

    def _alloc_fn(self) -> FunctionId:
        """Allocate a fresh ``FunctionId``."""
        fn_id = FunctionId(self._next_fn)
        self._next_fn += 1
        return fn_id

    def _prealloc_funcdef(self, funcdef: "FuncDef") -> None:
        """Pre-allocate SymbolId and FunctionId for a top-level FuncDef."""
        fn_id = self._alloc_fn()
        sym = self._alloc_sym(
            funcdef.node_id,
            name=funcdef.name,
            mutable=False,
            public=not funcdef.is_private,
            owner=ENTRY_ID,
        )
        self._fn_node_to_sym[funcdef.node_id] = sym
        self._fn_node_to_id[funcdef.node_id] = fn_id

    # Binder kinds whose values live in evaluation frames and can therefore be
    # captured by a closure (D5).  function_binding is resolved through the function
    # table; agent_binding/constructor_binding are not frame values in the M4 IR
    # (host prep / constructors are handled elsewhere) so they are not captures here.
    _CAPTURABLE_KINDS = frozenset({
        BinderKind.let_binding,
        BinderKind.var_binding,
        BinderKind.param_binding,
        BinderKind.catch_binder,
        BinderKind.pattern_binding,
    })

    def _pattern_binding_ids(self, pattern: Pattern, out: set[int]) -> None:
        """Collect node_ids of the variable binders a pattern introduces (D4 closed match)."""
        match pattern:
            case VarPattern():
                if pattern.node_id not in self._checked.resolved.bare_variant_patterns:
                    out.add(pattern.node_id)
            case ConstructorPattern():
                for pf in pattern.fields:
                    self._pattern_binding_ids(pf.pattern, out)
            case WildcardPattern() | LiteralPattern():
                pass
            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    def _record_capture(self, node_id: int, local_ids: set[int],
                        captured: dict[int, BindingRef]) -> None:
        ref = self._checked.resolved.resolution.get(node_id)
        if (ref is not None
                and ref.decl_node_id not in local_ids
                and ref.kind in self._CAPTURABLE_KINDS):
            captured[ref.decl_node_id] = ref

    def _scan_captures(self, node: Item, local_ids: set[int],
                       captured: dict[int, BindingRef]) -> None:
        """Single boundary-aware pass collecting a function body's free-var captures.

        Registers every in-body binder (let/var/pattern/catch) into ``local_ids``
        BEFORE descending the scope it governs, and records in ``captured`` every
        reference that resolves to a CAPTURABLE binding declared OUTSIDE the body.
        Stops at nested FuncDef/Lambda boundaries (they analyze their own free vars).
        A shared monotonically-growing ``local_ids`` set is safe: a reference can
        only resolve to a binder already in scope (declared earlier), so out-of-scope
        leakage into sibling branches can never produce a false local/false capture.
        """
        match node:
            case VarRef():
                self._record_capture(node.node_id, local_ids, captured)
            # Boundaries + declarations introduce no value captures for THIS function:
            case (Lambda() | FuncDef() | RecordDef() | EnumDef() | TypeAlias()
                  | ParamDecl() | ProgramDecl() | AgentDecl() | ConfigPragma() | ImportDecl()):
                return
            case LetDecl() | VarDecl():
                local_ids.add(node.node_id)
                self._scan_captures(node.value, local_ids, captured)
            case AssignStmt():
                # The assigned binding (NameTarget root, or IndexTarget root) must be
                # captured even when it is never otherwise read (e.g. `x := 5`).
                self._record_capture(node.node_id, local_ids, captured)
                if isinstance(node.target, IndexTarget):
                    self._scan_captures(node.target.obj, local_ids, captured)
                    self._scan_captures(node.target.index, local_ids, captured)
                self._scan_captures(node.value, local_ids, captured)
            case Block():
                for item in node.items:
                    self._scan_captures(item, local_ids, captured)
            case If():
                for br in node.branches:
                    if not isinstance(br.cond, ElseSentinel):
                        self._scan_captures(br.cond, local_ids, captured)
                    self._scan_captures(br.body, local_ids, captured)
            case Case():
                self._scan_captures(node.subject, local_ids, captured)
                for cbr in node.branches:
                    self._pattern_binding_ids(cbr.pattern, local_ids)
                    self._scan_captures(cbr.body, local_ids, captured)
            case Try():
                self._scan_captures(node.body, local_ids, captured)
                for clause in node.handlers:
                    if clause.binding is not None:
                        local_ids.add(clause.node_id)
                    self._scan_captures(clause.body, local_ids, captured)
            case Do():
                self._scan_captures(node.body, local_ids, captured)
                self._scan_captures(node.condition, local_ids, captured)
            case BinaryOp():
                self._scan_captures(node.left, local_ids, captured)
                self._scan_captures(node.right, local_ids, captured)
            case UnaryNot() | UnaryNeg():
                self._scan_captures(node.operand, local_ids, captured)
            case Cast() | IsTest():
                self._scan_captures(node.expr, local_ids, captured)
            case Call():
                self._scan_captures(node.callee, local_ids, captured)
                for arg in node.args:
                    self._scan_captures(arg, local_ids, captured)
                for na in node.named_args:
                    self._scan_captures(na.value, local_ids, captured)
            case FieldAccess():
                self._scan_captures(node.obj, local_ids, captured)
            case IndexAccess():
                self._scan_captures(node.obj, local_ids, captured)
                self._scan_captures(node.index, local_ids, captured)
            case ListLit():
                for elem in node.elements:
                    self._scan_captures(elem, local_ids, captured)
            case DictLit():
                for entry in node.entries:
                    self._scan_captures(entry.key, local_ids, captured)
                    self._scan_captures(entry.value, local_ids, captured)
            case Template():
                for seg in node.segments:
                    if isinstance(seg, InterpSegment):
                        self._scan_captures(seg.expr, local_ids, captured)
            case Raise():
                self._scan_captures(node.exc, local_ids, captured)
            case UnitLit() | IntLit() | DecimalLit() | BoolLit() | NullLit() | StringLit():
                pass
            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    def _compute_captures(
        self, funcdef: "FuncDef", param_decl_ids: "set[int]"
    ) -> "tuple[IrCapture, ...]":
        """Compute the captures for a FuncDef body using the single boundary-aware pass."""
        local_ids: set[int] = set(param_decl_ids)
        local_ids.add(funcdef.node_id)
        captured: dict[int, BindingRef] = {}
        self._scan_captures(funcdef.body, local_ids, captured)
        for param in funcdef.params:
            if param.default is not None:
                self._scan_captures(param.default, local_ids, captured)
        captures: list[IrCapture] = []
        for decl_id, ref in captured.items():
            sym = self._decl_to_sym.get(decl_id)
            assert sym is not None, (  # capturable outer bindings are always pre-allocated
                f"compiler bug: captured binding {ref.name!r} (decl_node_id={decl_id})"
                " has no allocated symbol"
            )
            captures.append(IrCapture(symbol=sym, by_cell=ref.mutable))
        return tuple(captures)

    def _lower_funcdef(self, funcdef: "FuncDef") -> IrExpr:
        """Lower a FuncDef to IrBind(sym, IrMakeClosure(fn_id, captures)).

        All top-level FuncDefs are pre-allocated before any body is lowered (phase 1),
        so the symbol + function-id are always present; nested ``def`` is rejected by
        the scope checker.
        """
        assert funcdef.node_id in self._fn_node_to_id, (
            f"compiler bug: FuncDef {funcdef.name!r} was not pre-allocated"
        )
        fn_id = self._fn_node_to_id[funcdef.node_id]
        fn_sym = self._fn_node_to_sym[funcdef.node_id]

        param_decl_ids: set[int] = set()
        param_syms: list[SymbolId] = []
        for param in funcdef.params:
            param_decl_ids.add(param.node_id)
            psym = self._alloc_sym(
                param.node_id,
                name=param.name,
                mutable=False,
                public=False,
                owner=fn_id,
            )
            param_syms.append(psym)

        captures = self._compute_captures(funcdef, param_decl_ids)

        ir_params: list[IrFunctionParam] = []
        for param, psym in zip(funcdef.params, param_syms):
            if param.default is not None:
                param_type_for_default = self._binding_type_for(param.node_id)
                default_ir: IrExpr | None = self.lower_coerced(
                    param.default, param_type_for_default
                )
            else:
                default_ir = None
            ir_params.append(IrFunctionParam(symbol=psym, default=default_ir))

        sig = self._checked.type_env.get_function_signature_by_node_id(funcdef.node_id)
        if sig is None:
            sig = self._checked.type_env.get_function_signature(funcdef.name)
        assert sig is not None, (
            f"compiler bug: no function signature for {funcdef.name!r}"
        )
        body_ir = self.lower_coerced(funcdef.body, sig.result)

        desc = FunctionDescriptor(
            function_id=fn_id,
            function_symbol=fn_sym,
            module_id=ENTRY_ID,
            params=tuple(ir_params),
            body=body_ir,
        )
        self._functions[fn_id] = desc

        loc = self._loc(funcdef.span)
        closure_ir = IrMakeClosure(location=loc, function_id=fn_id, captures=captures)
        return IrBind(location=loc, symbol=fn_sym, value=closure_ir)

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

    def _source_slice(self, span: SourceSpan) -> str:
        """Return the source-text covered by *span*.

        Mirrors ``Interpreter._source_slice`` in the legacy evaluator; used to
        capture the condition-expression source text for ``IrLoop.condition_source``
        (the ``MaxIterationsExceeded`` ``condition`` field).
        """
        return self._source_text[span.start_offset : span.end_offset]

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

            case Cast(expr=operand, test_only=test_only, span=span, node_id=nid):
                spec = self._checked.cast_specs[nid]
                source_type = self._node_type(operand.node_id)
                recipe = compile_recipe(source_type, spec.target_type, spec.kind)
                inner = self.lower_expr(operand)
                if not test_only:
                    return IrConvert(
                        location=self._loc(span),
                        value=inner,
                        recipe=recipe,
                        failure_mode=ConversionFailureMode.RAISE_CAST_ERROR,
                    )
                # `as?`: total casts always succeed — evaluate the (possibly
                # effectful) source, then yield True.  Fallible casts trial-convert.
                if spec.kind in (
                    CastKind.TOTAL_NOOP,
                    CastKind.TOTAL_RENDER,
                    CastKind.TOTAL_JSON,
                ):
                    return IrSequence(
                        location=self._loc(span),
                        items=(inner, IrConstBool(location=self._loc(span), value=True)),
                    )
                return IrConvert(
                    location=self._loc(span),
                    value=inner,
                    recipe=recipe,
                    failure_mode=ConversionFailureMode.RETURN_BOOL,
                )

            case IsTest(expr=operand, variant=variant, negated=negated, span=span):
                # The checker guarantees the operand is enum-typed (see
                # _check_is_test); build the nominal from its checked EnumType.
                operand_type = self._node_type(operand.node_id)
                assert isinstance(operand_type, EnumType), (
                    "is-test operand must be enum-typed (checker guarantees this)"
                )
                return IrVariantIs(
                    location=self._loc(span),
                    nominal=NominalId(operand_type.module_id, operand_type.name),
                    variant=variant,
                    value=self.lower_expr(operand),
                    negated=negated,
                )

            # ----------------------------------------------------------
            # Control flow — M3f-A: if, raise, try
            # ----------------------------------------------------------
            case If(branches=branches, span=span):
                return self._lower_if(branches, span)

            case Raise(exc=exc_expr, span=span):
                return IrRaise(
                    location=self._loc(span),
                    exc=self.lower_expr(exc_expr),
                )

            case Try(body=body_expr, handlers=handlers, span=span):
                return self._lower_try(body_expr, handlers, span)

            # ----------------------------------------------------------
            # Case expression — M3f-B
            # ----------------------------------------------------------
            case Case(subject=subject_expr, branches=branches, span=span):
                return self._lower_case(subject_expr, branches, span)

            # ----------------------------------------------------------
            # do…until loop → IrLoop (M3f-C)
            # ----------------------------------------------------------
            case Do(limit=loop_limit, body=body_expr, condition=cond_expr, span=span):
                condition_source = self._source_slice(cond_expr.span)
                return IrLoop(
                    location=self._loc(span),
                    limit=loop_limit,
                    body=self.lower_expr(body_expr),
                    condition=self.lower_expr(cond_expr),
                    condition_source=condition_source,
                )

            # ----------------------------------------------------------
            # Unsupported M4+ nodes — deferred
            # ----------------------------------------------------------
            case Lambda():
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
        are lowered to IrMakeRecord/IrMakeEnum/IrMakeException.  Direct user function
        calls are lowered to IrDirectCall.  All other calls (lambdas, indirect, host)
        still raise NotImplementedError (deferred M4b/M6).
        """
        callee = call_node.callee

        # Check for builtin calls first
        builtin_kind = self._checked.resolved.builtin_calls.get(nid)
        if builtin_kind is not None:
            raise NotImplementedError(
                "Lowering of builtin calls is not yet implemented (deferred to M6)"
            )

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
            # (c) Direct user function call
            if (
                callee_ref is not None
                and callee_ref.kind is BinderKind.function_binding
            ):
                return self._lower_direct_call(call_node, callee_ref, nid, span)

        elif isinstance(callee, FieldAccess):
            qcr = self._checked.resolved.qualified_constructor_refs.get(callee.node_id)
            if qcr is not None:
                owner_name, variant_name, _qcr_mid = qcr
                return self._lower_named_constructor_call(
                    nid, owner_name, variant_name, call_node.named_args, span
                )

        # Lambda/indirect/value call — deferred to M4b
        raise NotImplementedError(
            "Lowering of indirect/value calls is not yet implemented (deferred to M4b)"
        )

    def _lower_direct_call(
        self,
        call_node: "Call",
        callee_ref: BindingRef,
        result_node_id: int,
        span: "SourceSpan",
    ) -> IrDirectCall:
        """Lower a direct call to a named user function."""
        fn_id = self._fn_node_to_id.get(callee_ref.decl_node_id)
        assert fn_id is not None, (
            f"compiler bug: no FunctionId for function decl_node_id={callee_ref.decl_node_id!r}"
        )

        sig = self._checked.type_env.get_function_signature_by_node_id(callee_ref.decl_node_id)
        if sig is None:
            sig = self._checked.type_env.get_function_signature(callee_ref.name)
        assert sig is not None, (
            f"compiler bug: no signature for function {callee_ref.name!r}"
        )

        pos_args = list(call_node.args)
        named_args: dict[str, Expr] = {na.name: na.value for na in call_node.named_args}

        ir_args: list[IrExpr | UseDefault] = []
        pos_idx = 0
        for i, (pname, ptype, has_default) in enumerate(sig.params):
            if pname in named_args:
                ir_args.append(self.lower_coerced(named_args[pname], ptype))
            elif pos_idx < len(pos_args):
                ir_args.append(self.lower_coerced(pos_args[pos_idx], ptype))
                pos_idx += 1
            elif has_default:
                ir_args.append(UseDefault(param_index=i))
            else:
                raise AssertionError(
                    f"compiler bug: missing required arg for param {pname!r} in call to"
                    f" {callee_ref.name!r} (checker should have caught this)"
                )

        return IrDirectCall(
            location=self._loc(span),
            function_id=fn_id,
            arguments=tuple(ir_args),
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
        """Lower a ``Block``'s items to an ``IrBlock``.

        All items inside a block body are lowered as **nested** (``top_level=False``),
        so any ``let``/``var`` binders they declare are allocated with
        ``public=False`` and do not appear in ``_collect_results``.  Only the
        top-level module-initializer driver passes ``top_level=True``.
        """
        ir_items = tuple(self.lower_item(it, top_level=False) for it in items)
        # Filter out None (items with no runtime action); validate non-empty.
        real: list[IrExpr] = [x for x in ir_items if x is not None]
        assert real, "compiler bug: lowered block has no runtime items"
        return IrBlock(location=self._loc(span), items=tuple(real))

    # ------------------------------------------------------------------
    # Control-flow helpers (M3f-A)
    # ------------------------------------------------------------------

    def _lower_if(
        self,
        branches: "tuple[IfBranch, ...]",
        span: "SourceSpan",
    ) -> IrIf:
        """Lower an ``If`` AST node to ``IrIf``."""
        has_else = any(isinstance(br.cond, ElseSentinel) for br in branches)
        ir_branches: list[IrIfBranch] = []
        for branch in branches:
            if isinstance(branch.cond, ElseSentinel):
                ir_branches.append(
                    IrIfBranch(cond=None, body=self.lower_expr(branch.body))
                )
            else:
                ir_branches.append(
                    IrIfBranch(
                        cond=self.lower_expr(branch.cond),
                        body=self.lower_expr(branch.body),
                    )
                )
        return IrIf(
            location=self._loc(span),
            branches=tuple(ir_branches),
            has_else=has_else,
        )

    def _lower_try(
        self,
        body_expr: "Expr",
        handlers: "tuple[CatchClause, ...]",
        span: "SourceSpan",
    ) -> IrTry:
        """Lower a ``Try`` AST node to ``IrTry``."""
        return IrTry(
            location=self._loc(span),
            body=self.lower_expr(body_expr),
            handlers=tuple(self._lower_catch_clause(c) for c in handlers),
        )

    def _lower_catch_clause(self, clause: "CatchClause") -> IrCatchHandler:
        """Lower a ``CatchClause`` to an ``IrCatchHandler``."""
        # Determine nominal + display_name.
        exc_type = clause.exc_type
        if exc_type is None or exc_type == "_" or exc_type == "Exception":
            nominal: NominalId | None = None
            display_name: str | None = None
        else:
            nominal = NominalId(PRELUDE_ID, exc_type)
            display_name = exc_type

        # Allocate a SymbolId for the binding variable when present.
        # public=False: catch-clause binders are not top-level exported names.
        sym: SymbolId | None = None
        if clause.binding is not None:
            sym = self._alloc_sym(
                clause.node_id,
                name=clause.binding,
                mutable=False,
                public=False,
            )

        return IrCatchHandler(
            nominal=nominal,
            display_name=display_name,
            symbol=sym,
            body=self.lower_expr(clause.body),
        )

    # ------------------------------------------------------------------
    # Case expression helpers (M3f-B)
    # ------------------------------------------------------------------

    def _lower_case(
        self,
        subject_expr: "Expr",
        branches: "tuple[CaseBranch, ...]",
        span: "SourceSpan",
    ) -> IrCase:
        """Lower a ``Case`` AST node to ``IrCase`` with compiled match plans."""
        ir_arms = tuple(
            IrCaseArm(
                plan=self._compile_plan(branch.pattern),
                body=self.lower_expr(branch.body),
            )
            for branch in branches
        )
        return IrCase(
            location=self._loc(span),
            subject=self.lower_expr(subject_expr),
            arms=ir_arms,
        )

    def _compile_plan(self, pattern: Pattern) -> IrMatchPlan:
        """Compile a ``Pattern`` to a closed ``IrMatchPlan``.

        Closed ``match``/``assert_never`` dispatch over the ``Pattern`` union
        (D4) — mypy exhaustiveness makes a missing case a compile-time error.
        """
        match pattern:
            case WildcardPattern():
                return IrWildcardPlan()

            case VarPattern(name=name, node_id=nid):
                if nid in self._checked.resolved.bare_variant_patterns:
                    # Nullary constructor pattern — match by variant name, no binding.
                    return IrVariantPlan(variant=name)
                # Binder pattern — allocate a fresh SymbolId (private, public=False).
                sym = self._alloc_sym(nid, name=name, mutable=False, public=False)
                return IrBindPlan(symbol=sym)

            case LiteralPattern(literal=lit):
                # Lower the literal to its IrConst* node (re-use lower_expr).
                return IrLiteralPlan(value=self.lower_expr(lit))

            case ConstructorPattern(name=variant_name, fields=fields):
                field_plans: list[tuple[str, IrMatchPlan]] = [
                    (pf.name, self._compile_plan(pf.pattern)) for pf in fields
                ]
                return IrConstructorPlan(
                    variant=variant_name,
                    fields=tuple(field_plans),
                )

            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    # ------------------------------------------------------------------
    # Item lowering
    # ------------------------------------------------------------------

    def lower_item(self, item: Item, *, top_level: bool = False) -> IrExpr | None:
        """Lower a block item.

        Returns an ``IrExpr`` for nodes with runtime action, or ``None`` for
        purely compile-time declarations (type definitions, function defs, etc.)
        that have no IR representation in M2.

        The IrBlock construction must then filter out the ``None`` values.

        Parameters
        ----------
        top_level:
            When ``True``, ``let``/``var`` binders are allocated with
            ``public=True`` so they appear in ``_collect_results``.  When
            ``False`` (the default), binders are allocated with ``public=False``
            — they live in the flat per-invocation frame but are not exported.
            The module-initializer driver passes ``top_level=True``; all block
            bodies (``_lower_block``, if/try/case/do branch bodies) use the
            default ``False`` so future control-flow nodes inherit safe behaviour
            automatically.
        """
        match item:
            # ----------------------------------------------------------
            # Binders
            # ----------------------------------------------------------
            case LetDecl(name=name, value=rhs, span=span, node_id=nid):
                sym = self._alloc_sym(nid, name=name, mutable=False, public=top_level)
                binding_type = self._binding_type_for(nid)
                ir_val = self.lower_coerced(rhs, binding_type)
                return IrBind(location=self._loc(span), symbol=sym, value=ir_val)

            case VarDecl(name=name, value=rhs, span=span, node_id=nid):
                sym = self._alloc_sym(nid, name=name, mutable=True, public=top_level)
                binding_type = self._binding_type_for(nid)
                ir_val = self.lower_coerced(rhs, binding_type)
                return IrBind(location=self._loc(span), symbol=sym, value=ir_val)

            case AssignStmt(target=target, value=rhs, span=span, node_id=nid):
                return self._lower_assign(target, rhs, span, nid)

            # ----------------------------------------------------------
            # Declarations with no runtime action in M2
            # ----------------------------------------------------------
            case FuncDef() as funcdef:
                return self._lower_funcdef(funcdef)

            case (
                AgentDecl()
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
        self._build_nominals()

        body = self._checked.resolved.program.body

        # Phase 1: pre-allocate function symbols and IDs for mutual recursion
        for item in body.items:
            if isinstance(item, FuncDef):
                self._prealloc_funcdef(item)

        # Phase 2: lower all items
        ir_items: list[IrExpr] = []
        for item in body.items:
            ir = self.lower_item(item, top_level=True)
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
            functions=dict(self._functions),
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
