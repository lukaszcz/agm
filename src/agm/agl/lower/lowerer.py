"""Single-module lowerer for the AgL typeless execution IR.

Transforms a ``CheckedProgram`` into an ``ExecutableProgram`` for the supported
node subset.  Every implicit coercion is inserted explicitly at compile time via
``compile_coercion``; the evaluator switches only on pre-resolved ``Coercion``
descriptors and never inspects value types at runtime.

Supported AST nodes
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
      ProgramDecl, ConfigDecl, ImportDecl, ExportDecl

Any AST node outside this set raises ``NotImplementedError`` with a clear
message.  A missing checker side-table entry is a compiler bug and raises
``AssertionError``.

Dispatch uses structural ``match`` with a final ``assert_never`` arm.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import assert_never, cast

from agm.agl.ir.contracts import (
    ContractPayload,
    ContractRequest,
    ConversionFailureMode,
    DecodeSchema,
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
    IrBindPlan,
    IrBlock,
    IrBreak,
    IrCapture,
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
    IrIfBranch,
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
from agm.agl.ir.operations import (
    ArithKind,
    ArithOp,
    CmpOp,
    CompareKind,
    ContainsKind,
    IndexKind,
    IterKind,
    NumericKind,
    UnaryOp,
)
from agm.agl.ir.program import (
    DryRunEntry,
    ExecutableModule,
    ExecutableProgram,
    FunctionDescriptor,
    IrParam,
    NominalDescriptor,
    NominalKind,
    SourceFile,
    SymbolDescriptor,
    VariantDescriptor,
)
from agm.agl.ir.validate import validate_ir
from agm.agl.lower.coercions import compile_coercion
from agm.agl.lower.conversions import compile_recipe
from agm.agl.modules.ids import ENTRY_ID, PRELUDE_ID, STD_CORE_ID, ModuleId
from agm.agl.scope.symbols import BinderKind, BindingRef, BuiltinKind
from agm.agl.semantics.type_table import TypeTable
from agm.agl.semantics.types import (
    BUILTIN_EXCEPTIONS,
    BUILTIN_PRELUDE_TYPES,
    BoolType,
    BottomType,
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
    TypeVarType,
    UnitType,
    substitute,
)
from agm.agl.syntax.nodes import (
    AgentDecl,
    AssignStmt,
    AssignTarget,
    BinaryOp,
    BinOp,
    Block,
    BoolLit,
    Break,
    Call,
    Case,
    CaseBranch,
    Cast,
    CatchClause,
    ConfigDecl,
    ConstructorPattern,
    Continue,
    DecimalLit,
    DictLit,
    ElseSentinel,
    EnumDef,
    ExceptionDef,
    ExportDecl,
    Expr,
    FieldAccess,
    FuncDef,
    If,
    IfBranch,
    ImportDecl,
    IndexAccess,
    IndexTarget,
    InfixDecl,
    InterpSegment,
    IntLit,
    IsTest,
    Item,
    Lambda,
    LetDecl,
    ListLit,
    LiteralPattern,
    Loop,
    NamedArg,
    NameTarget,
    NullLit,
    Param,
    ParamDecl,
    Pattern,
    Placeholder,
    ProgramDecl,
    Raise,
    RecordDef,
    Return,
    StringLit,
    Template,
    TextSegment,
    Try,
    TypeAlias,
    TypeApply,
    UnaryNeg,
    UnaryNot,
    UnitLit,
    VarDecl,
    VarPattern,
    VarRef,
    WildcardPattern,
)
from agm.agl.syntax.spans import SourceSpan
from agm.agl.type_schema import (
    build_format_instructions,
    build_param_decoder,
    derive_schema_and_decode,
)
from agm.agl.typecheck.env import (
    CheckedProgram,
    FunctionSignature,
    OutputContractSpec,
    PartialCallSpec,
)
from agm.agl.typecheck.graph import CheckedModule
from agm.util.text import normalize_newlines

__all__ = ["_LinkState", "lower_program"]


def _contract_has_schema(
    spec: OutputContractSpec | None,
    payload: ContractPayload | None,
) -> bool:
    """Return whether a call-site contract carries a materialized schema."""
    return spec is not None and (
        spec.codec_name == "json" or (payload is not None and payload.json_schema is not None)
    )


def _add_builtin_nominals(
    nominals: dict[NominalId, NominalDescriptor], type_table: TypeTable
) -> None:
    """Register built-in prelude and exception nominal descriptors.

    Record/enum field and variant names, and exception field names, are all
    resolved through *type_table* (every built-in prelude and exception type
    is seeded into every table by ``create_seeded_type_table``).
    """
    for name, typ in BUILTIN_PRELUDE_TYPES.items():
        nominal = NominalId(PRELUDE_ID, name)
        if isinstance(typ, RecordType):
            nominals[nominal] = NominalDescriptor(
                nominal=nominal,
                display_name=name,
                kind=NominalKind.RECORD,
                fields=tuple(type_table.record_fields(typ).keys()),
                variants=(),
            )
            continue
        enum_type = cast(EnumType, typ)
        nominals[nominal] = NominalDescriptor(
            nominal=nominal,
            display_name=name,
            kind=NominalKind.ENUM,
            fields=(),
            variants=tuple(
                VariantDescriptor(vname, tuple(vfields.keys()))
                for vname, vfields in type_table.enum_variants(enum_type).items()
            ),
        )

    for exc_name, exc_type in BUILTIN_EXCEPTIONS.items():
        nominal = NominalId(PRELUDE_ID, exc_name)
        nominals[nominal] = NominalDescriptor(
            nominal=nominal,
            display_name=exc_name,
            kind=NominalKind.EXCEPTION,
            fields=tuple(type_table.exception_fields(exc_type).keys()),
            variants=(),
        )


# ---------------------------------------------------------------------------
# Internal lowerer state (one instance per lower_program call)
# ---------------------------------------------------------------------------


@dataclass
class _LinkState:
    next_sym: int = 0
    next_fn: int = 0
    next_source: int = 0
    next_contract: int = 0
    decl_to_sym: dict[int, SymbolId] = field(default_factory=dict)
    fn_node_to_sym: dict[int, SymbolId] = field(default_factory=dict)
    fn_node_to_id: dict[int, FunctionId] = field(default_factory=dict)
    symbols: dict[SymbolId, SymbolDescriptor] = field(default_factory=dict)
    functions: dict[FunctionId, FunctionDescriptor] = field(default_factory=dict)
    nominals: dict[NominalId, NominalDescriptor] = field(default_factory=dict)
    sources: dict[SourceId, SourceFile] = field(default_factory=dict)
    contracts: dict[ContractId, ContractRequest] = field(default_factory=dict)


_ARITH_OP_MAP: dict[BinOp, ArithOp] = {
    BinOp.ADD: ArithOp.ADD,
    BinOp.SUB: ArithOp.SUB,
    BinOp.MUL: ArithOp.MUL,
}

_CMP_OP_MAP: dict[BinOp, CmpOp] = {
    BinOp.LT: CmpOp.LT,
    BinOp.LE: CmpOp.LE,
    BinOp.GT: CmpOp.GT,
    BinOp.GE: CmpOp.GE,
}


class _Lowerer:
    """Holds all mutable state for one lowering pass."""

    def __init__(
        self,
        checked: CheckedProgram | CheckedModule,
        link: _LinkState,
        module_id: ModuleId,
        source_id: SourceId,
        source_text: str,
        contract_payloads: Mapping[int, ContractPayload] | None = None,
    ) -> None:
        self._checked = checked
        self._link = link
        self._module_id = module_id
        self._source_id = source_id
        self._source_text = normalize_newlines(source_text)
        self._params: list[IrParam] = []
        # Shared TypeTable built during checking; resolves record/enum field
        # and variant shapes for constructor lowering, nominal descriptors,
        # and contract/param schema derivation.
        self._type_table: TypeTable = checked.type_env.type_table
        self._contract_payloads: Mapping[int, ContractPayload] = (
            contract_payloads if contract_payloads is not None else {}
        )
        self._return_expected_stack: list[Type] = []

    @contextmanager
    def _return_context(self, expected: Type) -> Iterator[None]:
        """Track the coercion target for ``return`` inside a function body."""
        self._return_expected_stack.append(expected)
        try:
            yield
        finally:
            self._return_expected_stack.pop()

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
        sym = SymbolId(self._link.next_sym)
        self._link.next_sym += 1
        self._link.decl_to_sym[decl_node_id] = sym
        self._link.symbols[sym] = SymbolDescriptor(
            symbol_id=sym,
            mutable=mutable,
            public_name=name if public else None,
            owner=owner if owner is not None else self._module_id,
        )
        return sym

    def _alloc_synthetic_sym(
        self,
        *,
        mutable: bool,
        owner: "ModuleId | FunctionId | None" = None,
    ) -> SymbolId:
        """Allocate a fresh ``SymbolId`` for a lowering-internal synthetic binding.

        Unlike ``_alloc_sym``, this does NOT register an entry in ``decl_to_sym``
        (there is no AST declaration node).  ``public_name`` is ``None`` so the
        symbol is never exposed in ``_collect_results``.  Used for loop desugaring
        counters (``__count``, ``__n``) that must not be user-visible.
        """
        sym = SymbolId(self._link.next_sym)
        self._link.next_sym += 1
        self._link.symbols[sym] = SymbolDescriptor(
            symbol_id=sym,
            mutable=mutable,
            public_name=None,
            owner=owner if owner is not None else self._module_id,
        )
        return sym

    def _sym_for_decl(self, decl_node_id: int) -> SymbolId:
        """Return the pre-allocated ``SymbolId`` for a declaration node."""
        sym = self._link.decl_to_sym.get(decl_node_id)
        assert sym is not None, (
            f"compiler bug: no SymbolId for decl_node_id={decl_node_id!r}; "
            "declaration must be visited before its references"
        )
        return sym

    def _alloc_fn(self) -> FunctionId:
        """Allocate a fresh ``FunctionId``."""
        fn_id = FunctionId(self._link.next_fn)
        self._link.next_fn += 1
        return fn_id

    def _alloc_contract(self, request: ContractRequest) -> ContractId:
        """Allocate a fresh ContractId and register the ContractRequest."""
        cid = ContractId(self._link.next_contract)
        self._link.next_contract += 1
        self._link.contracts[cid] = request
        return cid

    def _contract_payload_for_spec(
        self, node_id: int, spec: OutputContractSpec
    ) -> tuple[str | None, str, DecodeSchema | None, tuple[tuple[str, DecodeSchema], ...]]:
        """Return JSON schema, instructions, decode root, and defs for a contract spec."""
        materialized = self._contract_payloads.get(node_id)
        if materialized is not None:
            return (
                materialized.json_schema,
                materialized.format_instructions,
                materialized.decode,
                materialized.defs,
            )
        if spec.codec_name != "json":
            return None, "", None, ()
        schema_dict, decode_plan = derive_schema_and_decode(spec.target_type, self._type_table)
        return (
            json.dumps(schema_dict),
            build_format_instructions(schema_dict),
            decode_plan.root,
            decode_plan.defs,
        )

    def _prealloc_funcdef(self, funcdef: "FuncDef") -> None:
        """Pre-allocate SymbolId and FunctionId for a top-level FuncDef."""
        fn_id = self._alloc_fn()
        sym = self._alloc_sym(
            funcdef.node_id,
            name=funcdef.name,
            mutable=False,
            public=not funcdef.is_private,
            owner=self._module_id,
        )
        self._link.fn_node_to_sym[funcdef.node_id] = sym
        self._link.fn_node_to_id[funcdef.node_id] = fn_id

    # Binder kinds whose values live in evaluation frames and can therefore be
    # captured by a closure.  function_binding is resolved through the function
    # table (via the base frame, which always contains all module-level bindings);
    # agent_binding/constructor_binding are not frame values in the IR
    # (host prep / constructors are handled elsewhere) so they are not captures here.
    _CAPTURABLE_KINDS = frozenset({
        BinderKind.let_binding,
        BinderKind.var_binding,
        BinderKind.param_binding,
        BinderKind.catch_binder,
        BinderKind.pattern_binding,
        BinderKind.agent_binding,
        BinderKind.loop_var_binding,
    })

    def _pattern_binding_ids(self, pattern: Pattern, out: set[int]) -> None:
        """Collect node_ids of the variable binders a pattern introduces."""
        match pattern:
            case VarPattern():
                if pattern.node_id not in self._checked.resolved.bare_variant_patterns:
                    out.add(pattern.node_id)
            case ConstructorPattern():
                for p in pattern.positional:
                    self._pattern_binding_ids(p, out)
                for pf in pattern.named:
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
            case (Lambda() | FuncDef() | RecordDef() | EnumDef() | ExceptionDef() | TypeAlias()
                  | ParamDecl() | ProgramDecl() | AgentDecl() | ConfigDecl() | ImportDecl()
                  | ExportDecl() | InfixDecl()):
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
            case Loop():
                if node.for_iter is not None:
                    self._scan_captures(node.for_iter, local_ids, captured)
                if node.for_range_to is not None:
                    self._scan_captures(node.for_range_to, local_ids, captured)
                if node.for_range_by is not None:
                    self._scan_captures(node.for_range_by, local_ids, captured)
                if node.bound is not None:
                    self._scan_captures(node.bound, local_ids, captured)
                if node.for_var is not None:
                    local_ids.add(node.node_id)
                if node.while_cond is not None:
                    self._scan_captures(node.while_cond, local_ids, captured)
                self._scan_captures(node.body, local_ids, captured)
                if node.until_cond is not None:
                    self._scan_captures(node.until_cond, local_ids, captured)
            case BinaryOp():
                self._scan_captures(node.left, local_ids, captured)
                self._scan_captures(node.right, local_ids, captured)
            case UnaryNot() | UnaryNeg():
                self._scan_captures(node.operand, local_ids, captured)
            case Cast() | IsTest():
                self._scan_captures(node.expr, local_ids, captured)
            case TypeApply():
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
            case Return():
                if node.value is not None:
                    self._scan_captures(node.value, local_ids, captured)
            case Break() | Continue() | Placeholder():
                pass  # leaf — no captures
            case UnitLit() | IntLit() | DecimalLit() | BoolLit() | NullLit() | StringLit():
                pass
            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    def _compute_captures_for(
        self,
        body: Expr,
        params: "tuple[Param, ...]",
        self_node_id: int,
        param_decl_ids: set[int],
    ) -> "tuple[IrCapture, ...]":
        """Compute the captures for a function body (FuncDef or Lambda).

        Scans *body* and each *params[i].default* for free-variable references
        that resolve to capturable bindings outside *param_decl_ids*.
        *self_node_id* is added to local_ids to prevent the function/lambda itself
        from being treated as a capture of itself.
        """
        local_ids: set[int] = set(param_decl_ids)
        local_ids.add(self_node_id)
        captured: dict[int, BindingRef] = {}
        self._scan_captures(body, local_ids, captured)
        for param in params:
            if param.default is not None:
                self._scan_captures(param.default, local_ids, captured)
        captures: list[IrCapture] = []
        for decl_id, ref in captured.items():
            sym = self._link.decl_to_sym.get(decl_id)
            assert sym is not None, (  # capturable outer bindings are always pre-allocated
                f"compiler bug: captured binding {ref.name!r} (decl_node_id={decl_id})"
                " has no allocated symbol"
            )
            # Module-owned bindings are resolved dynamically through frames[0].
            # Capturing them would snapshot module lets and require module vars to
            # exist before a top-level function closure can be hoisted.
            symbol_desc = self._link.symbols[sym]
            if (
                isinstance(symbol_desc.owner, ModuleId)
                and symbol_desc.public_name is not None
            ):
                continue
            captures.append(IrCapture(symbol=sym, by_cell=ref.mutable))
        return tuple(captures)

    def _compute_captures(
        self, funcdef: "FuncDef", param_decl_ids: "set[int]"
    ) -> "tuple[IrCapture, ...]":
        """Compute the captures for a FuncDef body using the single boundary-aware pass."""
        assert funcdef.body is not None, "builtin functions have no body"
        return self._compute_captures_for(
            funcdef.body, funcdef.params, funcdef.node_id, param_decl_ids
        )

    def _lower_funcdef(self, funcdef: "FuncDef") -> IrExpr:
        """Lower a FuncDef to IrBind(sym, IrMakeClosure(fn_id, captures)).

        All top-level FuncDefs are pre-allocated before any body is lowered (phase 1),
        so the symbol + function-id are always present; nested ``def`` is rejected by
        the scope checker.
        """
        assert not funcdef.is_builtin, "builtin functions are host-lowered at call sites"
        assert funcdef.body is not None, "builtin functions have no body"
        assert funcdef.node_id in self._link.fn_node_to_id, (
            f"compiler bug: FuncDef {funcdef.name!r} was not pre-allocated"
        )
        fn_id = self._link.fn_node_to_id[funcdef.node_id]
        fn_sym = self._link.fn_node_to_sym[funcdef.node_id]

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
                param_type_for_default = self._binding_type(param.node_id)
                default_ir: IrExpr | None = self.lower_coerced(
                    param.default, param_type_for_default
                )
            else:
                default_ir = None
            ir_params.append(IrFunctionParam(symbol=psym, default=default_ir))

        sig = self._checked.type_env.get_function_signature_by_node_id(funcdef.node_id)
        assert sig is not None, (
            f"compiler bug: no function signature for {funcdef.name!r}"
        )
        with self._return_context(sig.result):
            body_ir = self.lower_coerced(funcdef.body, sig.result)

        desc = FunctionDescriptor(
            function_id=fn_id,
            function_symbol=fn_sym,
            module_id=self._module_id,
            params=tuple(ir_params),
            body=body_ir,
            param_labels=tuple(repr(p.type) for p in sig.params),
            result_label=repr(sig.result),
        )
        self._link.functions[fn_id] = desc

        loc = self._loc(funcdef.span)
        closure_ir = IrMakeClosure(location=loc, function_id=fn_id, captures=captures)
        return IrBind(location=loc, symbol=fn_sym, value=closure_ir)

    def _lower_lambda(
        self,
        params: "tuple[Param, ...]",
        body_expr: Expr,
        span: "SourceSpan",
        node_id: int,
    ) -> IrMakeClosure:
        """Lower a Lambda expression to IrMakeClosure with a fresh FunctionDescriptor.

        Unlike a FuncDef, a lambda is an expression: the IrMakeClosure itself is the
        value; it is NOT wrapped in IrBind.  A fresh private function_symbol is
        allocated to satisfy the FunctionDescriptor + validator.

        Captures are computed relative to the lambda's own params (which become local
        to the lambda's frame) and body.  Capture-through (a lambda inside a def that
        captures the def's params) works automatically because IrMakeClosure is
        evaluated in the ENCLOSING frame (the def's call_frame at the time of the
        IrMakeClosure eval).
        """
        fn_id = self._alloc_fn()

        # Allocate a private symbol for the lambda's function_symbol entry.
        # This symbol is never loaded by name; it exists only to satisfy the
        # FunctionDescriptor.function_symbol field and the validator.
        fn_sym_node_id = node_id  # use the lambda node_id as the key
        fn_sym = self._alloc_sym(
            fn_sym_node_id,
            name=f"<lambda@{node_id}>",
            mutable=False,
            public=False,
            owner=fn_id,
        )

        # Allocate param SymbolIds.
        param_decl_ids: set[int] = set()
        param_syms: list[SymbolId] = []
        for param in params:
            param_decl_ids.add(param.node_id)
            psym = self._alloc_sym(
                param.node_id,
                name=param.name,
                mutable=False,
                public=False,
                owner=fn_id,
            )
            param_syms.append(psym)

        captures = self._compute_captures_for(body_expr, params, node_id, param_decl_ids)

        # Get the full FunctionType (checker records it on the lambda's node_id).
        fn_type = self._node_type(node_id)
        assert isinstance(fn_type, FunctionType), (
            f"compiler bug: Lambda node {node_id!r} has non-FunctionType node_type {fn_type!r}"
        )

        # Build IR params; lower param defaults.
        #
        # Unlike funcdef params, lambda param defaults are NOT type-checked
        # by the checker (``_check_lambda`` skips defaults), so their
        # ``node_type`` entries are absent.  Use ``lower_expr`` directly to
        # avoid an AssertionError from ``_node_type``.  The default expression
        # is still required to be type-compatible (guaranteed by the checker
        # when the funcdef path is taken; for lambdas the type annotation on
        # the param already pins the type).
        ir_params: list[IrFunctionParam] = []
        for param, psym in zip(params, param_syms):
            if param.default is not None:
                default_ir: IrExpr | None = self.lower_expr(param.default)
            else:
                default_ir = None
            ir_params.append(IrFunctionParam(symbol=psym, default=default_ir))

        # Lower the body coerced to the declared return type (bakes result coercion in).
        with self._return_context(fn_type.result):
            body_ir = self.lower_coerced(body_expr, fn_type.result)

        desc = FunctionDescriptor(
            function_id=fn_id,
            function_symbol=fn_sym,
            module_id=self._module_id,
            params=tuple(ir_params),
            body=body_ir,
            param_labels=tuple(repr(param_type) for param_type in fn_type.params),
            result_label=repr(fn_type.result),
        )
        self._link.functions[fn_id] = desc

        loc = self._loc(span)
        return IrMakeClosure(location=loc, function_id=fn_id, captures=captures)

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

        Used to capture expression source text for runtime diagnostics, such as
        the ``condition`` field of ``MaxIterationsExceeded`` built during loop
        desugaring.
        """
        return self._source_text[span.start_offset : span.end_offset]

    # ------------------------------------------------------------------
    # Binding-type helpers
    # ------------------------------------------------------------------

    def _binding_type(self, decl_node_id: int) -> Type:
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

    def _match_typevars(self, template: Type, concrete: Type, subst: dict[str, Type]) -> None:
        """Best-effort one-sided match used to recover erased generic call types."""
        if isinstance(template, TypeVarType):
            subst.setdefault(template.name, concrete)
            return
        if isinstance(template, ListType) and isinstance(concrete, ListType):
            self._match_typevars(template.elem, concrete.elem, subst)
        elif isinstance(template, DictType) and isinstance(concrete, DictType):
            self._match_typevars(template.value, concrete.value, subst)
        elif isinstance(template, FunctionType) and isinstance(concrete, FunctionType):
            if len(template.params) == len(concrete.params):
                for template_param, concrete_param in zip(template.params, concrete.params):
                    self._match_typevars(template_param, concrete_param, subst)
                self._match_typevars(template.result, concrete.result, subst)
        elif (
            isinstance(template, (RecordType, EnumType))
            and isinstance(concrete, (RecordType, EnumType))
            and type(template) is type(concrete)
            and template.name == concrete.name
            and len(template.type_args) == len(concrete.type_args)
        ):
            for template_arg, concrete_arg in zip(template.type_args, concrete.type_args):
                self._match_typevars(template_arg, concrete_arg, subst)

    # ------------------------------------------------------------------
    # Core lowering with optional coercion wrapping
    # ------------------------------------------------------------------

    def _coerce_ir(
        self,
        ir: IrExpr,
        source: Type,
        expected: Type,
        location: Location,
    ) -> IrExpr:
        """Wrap pre-lowered IR in ``IrCoerce`` when ``source`` needs ``expected``."""
        op = compile_coercion(source, expected, self._type_table)
        if op is None:
            return ir
        return IrCoerce(location=location, value=ir, operation=op)

    def lower_coerced(self, node: Expr, expected: Type) -> IrExpr:
        """Lower *node* as an expression, then wrap in ``IrCoerce`` if needed.

        The node's own checked type is retrieved via ``node_types``; a coercion
        from that type to *expected* is compiled and, if non-``None``, wraps the
        result in ``IrCoerce``.
        """
        ir = self.lower_expr(node)
        own_type = self._node_type(node.node_id)
        return self._coerce_ir(ir, own_type, expected, self._loc(node.span))

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
                qcr = self._checked.resolved.qualified_constructor_refs.get(nid)
                if qcr is not None:
                    owner_name, variant_name, qcr_mid = qcr
                    node_typ = self._node_type(nid)
                    if isinstance(node_typ, FunctionType):
                        nominal, display = self._nominal_for_cref_owner(owner_name, qcr_mid)
                        return IrMakeConstructor(
                            location=self._loc(span),
                            nominal=nominal,
                            display_name=display,
                            variant=variant_name,
                        )
                    return self._lower_nullary_constructor(nid, owner_name, variant_name, span)

                # Check for constructor reference FIRST (mirrors legacy _eval_var_ref).
                cref = self._checked.resolved.constructor_refs.get(nid)
                if cref is not None:
                    node_typ = self._node_type(nid)
                    if isinstance(node_typ, FunctionType):
                        # Constructor with fields used as a value → IrMakeConstructor.
                        binding = self._checked.resolved.resolution.get(nid)
                        owner_module = binding.module_id if binding is not None else None
                        nominal, display = self._nominal_for_cref_owner(
                            cref.owner_name, owner_module
                        )
                        return IrMakeConstructor(
                            location=self._loc(span),
                            nominal=nominal,
                            display_name=display,
                            variant=cref.variant,
                        )
                    # Nullary constructor used as a value → construct immediately.
                    # AgL grammar requires ≥1 field in a record, so a nullary
                    # record VarRef is impossible under the current grammar.
                    return self._lower_nullary_constructor(
                        nid, cref.owner_name, cref.variant, span
                    )

                ref = self._checked.resolved.resolution.get(nid)
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
            # Operator nodes
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
            # Field access → IrField
            # ----------------------------------------------------------
            case FieldAccess(obj=obj_expr, field=field_name, span=span):
                return IrField(
                    location=self._loc(span),
                    value=self.lower_expr(obj_expr),
                    field=field_name,
                )

            # ----------------------------------------------------------
            # Index access → IrIndex
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
            # Template string → IrRenderTemplate
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
            # Call — constructor calls lowered here; all other calls deferred
            # ----------------------------------------------------------
            case Call(node_id=nid, span=span) as call_node:
                return self._lower_call(call_node, nid, span)

            case TypeApply(expr=applied_expr):
                return self.lower_expr(applied_expr)

            case Cast(expr=operand, test_only=test_only, span=span, node_id=nid):
                spec = self._checked.cast_specs[nid]
                source_type = self._node_type(operand.node_id)
                recipe = compile_recipe(
                    source_type, spec.target_type, spec.kind, self._type_table
                )
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
            # Control flow — if, raise, try
            # ----------------------------------------------------------
            case If(branches=branches, span=span, node_id=nid):
                return self._lower_if(branches, span, self._node_type(nid))

            case Raise(exc=exc_expr, span=span):
                return IrRaise(
                    location=self._loc(span),
                    exc=self.lower_expr(exc_expr),
                )

            case Return(value=value_expr, span=span):
                assert self._return_expected_stack, (
                    "compiler bug: return lowered outside a function"
                )
                expected = self._return_expected_stack[-1]
                return IrReturn(
                    location=self._loc(span),
                    value=IrConstUnit(location=self._loc(span))
                    if value_expr is None
                    else self.lower_coerced(value_expr, expected),
                )

            case Try(body=body_expr, handlers=handlers, span=span):
                return self._lower_try(body_expr, handlers, span)

            # ----------------------------------------------------------
            # Case expression
            # ----------------------------------------------------------
            case Case(subject=subject_expr, branches=branches, span=span, node_id=nid):
                return self._lower_case(
                    subject_expr, branches, span, self._node_type(nid)
                )

            # ----------------------------------------------------------
            # loop expression → IrLoop (desugared by _lower_loop)
            # ----------------------------------------------------------
            case Loop(
                for_var=for_var,
                for_iter=for_iter_expr,
                for_range_to=for_range_to_expr,
                for_range_down=for_range_down,
                for_range_by=for_range_by_expr,
                while_cond=while_cond_expr,
                bound=bound_expr,
                body=body_expr,
                until_cond=until_cond_expr,
                span=span,
                node_id=loop_nid,
            ):
                return self._lower_loop(
                    for_var=for_var,
                    for_iter_expr=for_iter_expr,
                    for_range_to_expr=for_range_to_expr,
                    for_range_down=for_range_down,
                    for_range_by_expr=for_range_by_expr,
                    while_cond_expr=while_cond_expr,
                    bound_expr=bound_expr,
                    body_expr=body_expr,
                    until_cond_expr=until_cond_expr,
                    span=span,
                    loop_nid=loop_nid,
                )

            # ----------------------------------------------------------
            # break / continue — wire to the existing IR signals
            # ----------------------------------------------------------
            case Break(span=span):
                return IrBreak(location=self._loc(span))

            case Continue(span=span):
                return IrContinue(location=self._loc(span))

            # ----------------------------------------------------------
            # Lambda expression
            # ----------------------------------------------------------
            case Lambda(params=params, body=body_expr, span=span, node_id=nid):
                return self._lower_lambda(params, body_expr, span, nid)

            case Placeholder():
                raise AssertionError("placeholder reached lowering outside a partial call")

            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    # ------------------------------------------------------------------
    # Loop desugar
    # ------------------------------------------------------------------

    def _lower_loop(
        self,
        *,
        for_var: "str | None",
        for_iter_expr: "Expr | None",
        for_range_to_expr: "Expr | None",
        for_range_down: bool,
        for_range_by_expr: "Expr | None",
        while_cond_expr: "Expr | None",
        bound_expr: "Expr | None",
        body_expr: "Expr",
        until_cond_expr: "Expr | None",
        span: "SourceSpan",
        loop_nid: int,
    ) -> IrExpr:
        """Desugar a ``Loop`` AST node to ``IrLoop(body)``.

        **Collection for — pre-loop** (emitted as needed):
        - ``IrBind(__it, IrIterInit(kind, lower(for_iter_expr)))``  — mutable; for ``for``
        - ``IrBind(__n, lower_coerced(bound_expr, IntType()))``      — immutable; for bound
        - ``IrBind(__count, IrConstInt(0))``                         — mutable; for bound

        **Integer-range for — pre-loop** (when ``for_range_to_expr`` is not ``None``):
        - ``IrBind(__cur, lower_coerced(for_iter_expr, IntType()))`` — mutable cursor (start ``a``)
        - ``IrBind(__end, lower_coerced(for_range_to_expr, IntType()))`` — immutable (bound ``b``)
        - ``IrBind(__step, lower_coerced(for_range_by_expr, IntType()))`` or ``IrConstInt(1)``
        - ``IrIf(step <= 0 => IrRaise(RangeError))`` — step guard
        - (followed by the optional ``[n]`` bound items as for collection)

        **``IrLoop(body = IrBlock([…]))``** with:
        For collection ``for``:
        1. For exhaustion check  (``if not IrIterHasNext(__it) => break``)
        2. For variable bind     (``let for_var = IrIterNext(__it)``)
        For integer-range ``for``:
        1. Range termination check  (``if __cur > __end`` or ``__cur < __end`` => break)
        2a. Loop variable bind      (``let for_var = IrLoad(__cur)``)
        2b. Cursor advance          (``__cur := __cur + __step`` or ``… - …``)
        Shared items 3–7:
        3. While guard           (only if ``while_cond_expr`` is not ``None``)
        4. Bound check           (only if bounded)
        5. Count increment       (only if bounded)
        6. Body                  (always)
        7. Until guard           (only if ``until_cond_expr`` is not ``None``)

        Returns ``IrSequence(pre_items..., IrLoop)`` when pre-loop items exist;
        the plain ``IrLoop`` when not.
        """
        loc = self._loc(span)
        pre_items: list[IrExpr] = []
        it_sym: SymbolId | None = None
        cur_sym: SymbolId | None = None
        end_sym: SymbolId | None = None
        step_sym: SymbolId | None = None
        n_sym: SymbolId | None = None
        count_sym: SymbolId | None = None

        if for_range_to_expr is not None:
            # ---- Integer-range for pre-loop ----
            # assert for_iter_expr is not None is guaranteed by the parser/typechecker
            assert for_iter_expr is not None  # lower bound is in for_iter
            # __cur: mutable int cursor, initialised to start a
            cur_sym = self._alloc_synthetic_sym(mutable=True)
            pre_items.append(
                IrBind(
                    location=loc,
                    symbol=cur_sym,
                    value=self.lower_coerced(for_iter_expr, IntType()),
                )
            )
            # __end: immutable int, the to/downto bound b
            end_sym = self._alloc_synthetic_sym(mutable=False)
            pre_items.append(
                IrBind(
                    location=loc,
                    symbol=end_sym,
                    value=self.lower_coerced(for_range_to_expr, IntType()),
                )
            )
            # __step: immutable int; from by-expr or default 1
            step_sym = self._alloc_synthetic_sym(mutable=False)
            step_value: IrExpr
            if for_range_by_expr is not None:
                step_value = self.lower_coerced(for_range_by_expr, IntType())
            else:
                step_value = IrConstInt(location=loc, value=1)
            pre_items.append(IrBind(location=loc, symbol=step_sym, value=step_value))
            # Step guard: if __step <= 0 => raise RangeError(...)
            pre_items.append(
                IrIf(
                    location=loc,
                    branches=(
                        IrIfBranch(
                            cond=IrCompare(
                                location=loc,
                                op=CmpOp.LE,
                                kind=CompareKind.INT,
                                lhs=IrLoad(location=loc, symbol=step_sym),
                                rhs=IrConstInt(location=loc, value=0),
                            ),
                            body=IrRaise(
                                location=loc,
                                exc=IrMakeException(
                                    location=loc,
                                    nominal=NominalId(PRELUDE_ID, "RangeError"),
                                    display_name="RangeError",
                                    fields=(
                                        (
                                            "message",
                                            IrConstText(
                                                location=loc,
                                                value="loop step must be positive",
                                            ),
                                        ),
                                        ("trace_id", AutoTraceField()),
                                    ),
                                ),
                            ),
                        ),
                    ),
                    has_else=False,
                )
            )
        else:
            # ---- Collection for pre-loop ----
            if for_iter_expr is not None:
                iter_type = self._node_type(for_iter_expr.node_id)
                if isinstance(iter_type, ListType):
                    iter_kind = IterKind.LIST
                elif isinstance(iter_type, DictType):
                    iter_kind = IterKind.DICT_KEYS
                else:  # TextType
                    iter_kind = IterKind.TEXT
                it_sym = self._alloc_synthetic_sym(mutable=True)
                pre_items.append(
                    IrBind(
                        location=loc,
                        symbol=it_sym,
                        value=IrIterInit(
                            location=loc,
                            kind=iter_kind,
                            collection=self.lower_expr(for_iter_expr),
                        ),
                    )
                )

        # Allocate for_var symbol in the loop frame (immutable let-by-value)
        for_var_sym: SymbolId | None = None
        if for_var is not None:
            for_var_sym = self._alloc_sym(loop_nid, name=for_var, mutable=False, public=False)

        # Pre-loop: bound initialisation (shared by both paths)
        if bound_expr is not None:
            n_sym = self._alloc_synthetic_sym(mutable=False)
            pre_items.append(
                IrBind(
                    location=loc,
                    symbol=n_sym,
                    value=self.lower_coerced(bound_expr, IntType()),
                )
            )
            count_sym = self._alloc_synthetic_sym(mutable=True)
            pre_items.append(
                IrBind(
                    location=loc,
                    symbol=count_sym,
                    value=IrConstInt(location=loc, value=0),
                )
            )

        body_items: list[IrExpr] = []

        if cur_sym is not None:
            # ---- Integer-range body items 1 and 2 ----
            assert end_sym is not None
            assert step_sym is not None
            # Item 1: range termination — if __cur > __end (to) or __cur < __end (downto) => break
            term_op = CmpOp.LT if for_range_down else CmpOp.GT
            body_items.append(
                IrIf(
                    location=loc,
                    branches=(
                        IrIfBranch(
                            cond=IrCompare(
                                location=loc,
                                op=term_op,
                                kind=CompareKind.INT,
                                lhs=IrLoad(location=loc, symbol=cur_sym),
                                rhs=IrLoad(location=loc, symbol=end_sym),
                            ),
                            body=IrBreak(location=loc),
                        ),
                    ),
                    has_else=False,
                )
            )
            # Item 2a: bind loop variable to current cursor value (read before advance).
            # A range for always has a loop variable (guaranteed by the parser).
            assert for_var_sym is not None, "compiler bug: range for has no loop variable symbol"
            body_items.append(
                IrBind(
                    location=loc,
                    symbol=for_var_sym,
                    value=IrLoad(location=loc, symbol=cur_sym),
                )
            )
            # Item 2b: advance cursor — __cur := __cur + __step (to) or __cur - __step (downto)
            advance_op = ArithOp.SUB if for_range_down else ArithOp.ADD
            body_items.append(
                IrAssign(
                    location=loc,
                    symbol=cur_sym,
                    path=(),
                    value=IrArith(
                        location=loc,
                        op=advance_op,
                        kind=ArithKind.INT,
                        lhs=IrLoad(location=loc, symbol=cur_sym),
                        rhs=IrLoad(location=loc, symbol=step_sym),
                    ),
                )
            )
        else:
            # ---- Collection for body items 1 and 2 ----
            # Item 1: for exhaustion check — if not IrIterHasNext(__it) then break
            if it_sym is not None:
                body_items.append(
                    IrIf(
                        location=loc,
                        branches=(
                            IrIfBranch(
                                cond=IrUnary(
                                    location=loc,
                                    op=UnaryOp.NOT,
                                    kind=None,
                                    value=IrIterHasNext(
                                        location=loc,
                                        iterator=IrLoad(location=loc, symbol=it_sym),
                                    ),
                                ),
                                body=IrBreak(location=loc),
                            ),
                        ),
                        has_else=False,
                    )
                )

            # Item 2: for variable bind — let for_var = IrIterNext(__it)
            if it_sym is not None and for_var_sym is not None:
                body_items.append(
                    IrBind(
                        location=loc,
                        symbol=for_var_sym,
                        value=IrIterNext(
                            location=loc,
                            iterator=IrLoad(location=loc, symbol=it_sym),
                        ),
                    )
                )

        # Item 3: while guard — if not while_cond then break
        if while_cond_expr is not None:
            body_items.append(
                IrIf(
                    location=loc,
                    branches=(
                        IrIfBranch(
                            cond=IrUnary(
                                location=loc,
                                op=UnaryOp.NOT,
                                kind=None,
                                value=self.lower_coerced(while_cond_expr, BoolType()),
                            ),
                            body=IrBreak(location=loc),
                        ),
                    ),
                    has_else=False,
                )
            )

        # Item 4: bound check — if __count >= __n => inner_if
        if n_sym is not None and count_sym is not None:
            until_source = (
                self._source_slice(until_cond_expr.span)
                if until_cond_expr is not None
                else "false"
            )
            # Inner if: if __count == 0 => IrBreak else => IrRaise(MaxIterationsExceeded)
            inner_if = IrIf(
                location=loc,
                branches=(
                    IrIfBranch(
                        cond=IrCompare(
                            location=loc,
                            op=CmpOp.EQ,
                            kind=CompareKind.STRUCTURAL,
                            lhs=IrLoad(location=loc, symbol=count_sym),
                            rhs=IrConstInt(location=loc, value=0),
                        ),
                        body=IrBreak(location=loc),
                    ),
                    IrIfBranch(
                        cond=None,
                        body=IrRaise(
                            location=loc,
                            exc=IrMakeException(
                                location=loc,
                                nominal=NominalId(PRELUDE_ID, "MaxIterationsExceeded"),
                                display_name="MaxIterationsExceeded",
                                fields=(
                                    (
                                        "message",
                                        IrRenderTemplate(
                                            location=loc,
                                            segments=(
                                                IrTemplateText("Loop exhausted after "),
                                                IrTemplateValue(
                                                    IrLoad(location=loc, symbol=n_sym)
                                                ),
                                                IrTemplateText(" iterations"),
                                            ),
                                        ),
                                    ),
                                    ("trace_id", AutoTraceField()),
                                    ("limit", IrLoad(location=loc, symbol=n_sym)),
                                    (
                                        "condition",
                                        IrConstText(location=loc, value=until_source),
                                    ),
                                    (
                                        "last_condition_value",
                                        IrConstBool(location=loc, value=False),
                                    ),
                                    ("metadata", IrConstJsonNull(location=loc)),
                                ),
                            ),
                        ),
                    ),
                ),
                has_else=True,
            )
            # Outer if: if __count >= __n => inner_if   (has_else=False → yields unit)
            body_items.append(
                IrIf(
                    location=loc,
                    branches=(
                        IrIfBranch(
                            cond=IrCompare(
                                location=loc,
                                op=CmpOp.GE,
                                kind=CompareKind.INT,
                                lhs=IrLoad(location=loc, symbol=count_sym),
                                rhs=IrLoad(location=loc, symbol=n_sym),
                            ),
                            body=inner_if,
                        ),
                    ),
                    has_else=False,
                )
            )

            # Item 5: count increment — __count := __count + 1
            body_items.append(
                IrAssign(
                    location=loc,
                    symbol=count_sym,
                    path=(),
                    value=IrArith(
                        location=loc,
                        op=ArithOp.ADD,
                        kind=ArithKind.INT,
                        lhs=IrLoad(location=loc, symbol=count_sym),
                        rhs=IrConstInt(location=loc, value=1),
                    ),
                )
            )

        # Item 6: body (value discarded)
        body_items.append(self.lower_expr(body_expr))

        # Item 7: until guard (only when until_cond is present)
        if until_cond_expr is not None:
            body_items.append(
                IrIf(
                    location=loc,
                    branches=(
                        IrIfBranch(
                            cond=self.lower_coerced(until_cond_expr, BoolType()),
                            body=IrBreak(location=loc),
                        ),
                    ),
                    has_else=False,
                )
            )

        # A loop is self-bounded (guarded) when it has a [n] bound (which raises
        # MaxIterationsExceeded itself) or a for clause (bounded by a finite
        # collection).  The host's global max-iters safety valve applies only
        # to unguarded loops, so a large do[n] or a for over a big collection
        # is never cut short by the host safety net.
        guarded = bound_expr is not None or for_var is not None
        loop = IrLoop(
            location=loc,
            body=IrBlock(location=loc, items=tuple(body_items)),
            guarded=guarded,
        )

        if pre_items:
            return IrSequence(location=loc, items=(*pre_items, loop))
        return loop

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
    # Operator helpers
    # ------------------------------------------------------------------

    def _lower_binary_op(
        self, op: BinOp, left: Expr, right: Expr, span: SourceSpan
    ) -> IrExpr:
        """Lower a BinaryOp to the appropriate IR node."""
        loc = self._loc(span)

        left_type = self._node_type(left.node_id)
        if isinstance(left_type, BottomType):
            return self.lower_expr(left)

        if op is BinOp.AND:
            return IrAnd(location=loc, lhs=self.lower_expr(left), rhs=self.lower_expr(right))
        if op is BinOp.OR:
            return IrOr(location=loc, lhs=self.lower_expr(left), rhs=self.lower_expr(right))

        right_type = self._node_type(right.node_id)
        if isinstance(right_type, BottomType):
            return IrSequence(location=loc, items=(self.lower_expr(left), self.lower_expr(right)))
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
        arith_op = _ARITH_OP_MAP[op]
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
        cmp_op = _CMP_OP_MAP[op]
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
    # Constructor lowering helpers
    # ------------------------------------------------------------------

    def _nominal_for_cref_owner(
        self, owner_name: str, owner_module: ModuleId | None = None
    ) -> tuple[NominalId, str]:
        """Return (NominalId, display_name) for a constructor owner by name.

        Uses the resolved type's own ``module_id`` (which equals ``ENTRY_ID``
        for single-module programs and ``PRELUDE_ID`` for built-ins).
        """
        typ = (
            self._checked.type_env.resolve_type_by_module_id(owner_module, owner_name)
            if owner_module is not None and not owner_module.is_entry
            else self._checked.type_env.get_type(owner_name)
        )
        if isinstance(typ, RecordType):
            return NominalId(typ.module_id, typ.name), typ.name
        if isinstance(typ, EnumType):
            return NominalId(typ.module_id, typ.name), typ.name
        if isinstance(typ, ExceptionType):  # pragma: no cover
            # Exception constructors as first-class values are rejected by the checker.
            return NominalId(typ.module_id, typ.name), typ.name
        # Fallback for generic types: get from GenericTypeDef template.  # pragma: no cover
        gdef = (
            self._checked.type_env.get_generic_type_from_module(owner_module, owner_name)
            if owner_module is not None and not owner_module.is_entry
            else self._checked.type_env.get_generic_type(owner_name)
        )
        if gdef is None:
            imported = self._checked.type_env.get_open_imported_generic_type(owner_name)
            if imported is not None:
                imported_module, imported_name, gdef = imported
                owner_module = imported_module
                owner_name = imported_name
        if gdef is not None:  # pragma: no cover
            tmpl = gdef.template  # pragma: no cover
            if isinstance(tmpl, RecordType):  # pragma: no cover
                return NominalId(tmpl.module_id, owner_name), owner_name  # pragma: no cover
            if isinstance(tmpl, EnumType):  # pragma: no cover
                return NominalId(tmpl.module_id, owner_name), owner_name  # pragma: no cover
        return NominalId(ENTRY_ID, owner_name), owner_name  # pragma: no cover

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

    def _lower_builtin_call(
        self,
        kind: BuiltinKind,
        call_node: "Call",
        span: "SourceSpan",
    ) -> IrExpr:
        """Lower a builtin call node by dispatching on ``BuiltinKind``.

        Host builtins (``PRINT``, ``PARSE_JSON``, ``ASK``, ``ASK_REQUEST``,
        and ``EXEC``) are lowered here.
        """
        loc = self._loc(span)
        match kind:
            case BuiltinKind.PRINT:
                # print(expr) — lower the single positional argument.
                arg_ir = self.lower_expr(call_node.args[0])
                return IrPrint(location=loc, value=arg_ir)

            case BuiltinKind.RENDER:
                # render(expr, pretty:, quote_strings:) — lower the value and
                # any supplied boolean display options.
                arg_ir = self.lower_expr(call_node.args[0])
                pretty = None
                quote_strings = None
                for named in call_node.named_args:
                    lowered = self.lower_expr(named.value)
                    if named.name == "pretty":
                        pretty = lowered
                    else:
                        quote_strings = lowered
                return IrRenderValue(
                    location=loc,
                    value=arg_ir,
                    pretty=pretty,
                    quote_strings=quote_strings,
                )

            case BuiltinKind.PARSE_JSON:
                # parse_json(text) — arg is statically text; lower without coercion.
                arg_ir = self.lower_expr(call_node.args[0])
                return IrParseJson(location=loc, value=arg_ir)

            case BuiltinKind.ASK:
                return self._lower_ask_call(call_node, span, structured_exec=False)

            case BuiltinKind.ASK_REQUEST:
                return self._lower_ask_call(call_node, span, structured_exec=False, is_request=True)

            case BuiltinKind.EXEC:
                return self._lower_exec_call(call_node, span)

            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    def _lower_call(self, call_node: "Call", nid: int, span: "SourceSpan") -> IrExpr:
        """Lower a Call node.

        Constructor calls (VarRef callee resolving to a constructor) are lowered
        to IrMakeRecord/IrMakeEnum/IrMakeException.  Direct user function
        calls are lowered to IrDirectCall.  Lambda calls are lowered to IrMakeClosure,
        indirect calls to IrIndirectCall, and host builtins to
        IrPrint/IrRenderValue/IrParseJson/IrAsk/IrAskRequest/IrExec.
        """
        callee = call_node.callee

        partial_spec = self._checked.partial_calls.get(nid)
        if partial_spec is not None:
            return self._lower_partial_call(call_node, partial_spec, span)

        # Check for builtin calls first
        builtin_kind = self._checked.resolved.builtin_calls.get(nid)
        if builtin_kind is not None:
            return self._lower_builtin_call(builtin_kind, call_node, span)

        # (a) VarRef callee in constructor_refs
        if isinstance(callee, VarRef):
            qcr = self._checked.resolved.qualified_constructor_refs.get(callee.node_id)
            if qcr is not None:
                owner_name, variant_name, _qcr_mid = qcr
                return self._lower_named_constructor_call(
                    nid,
                    owner_name,
                    variant_name,
                    span,
                )
            cref = self._checked.resolved.constructor_refs.get(callee.node_id)
            if cref is not None:
                return self._lower_named_constructor_call(
                    nid,
                    cref.owner_name,
                    cref.variant,
                    span,
                )
            # (b) VarRef callee resolving via BinderKind.constructor_binding.
            callee_ref = self._checked.resolved.resolution.get(callee.node_id)
            if (
                callee_ref is not None
                and callee_ref.kind is BinderKind.constructor_binding
            ):
                return self._lower_named_constructor_call(
                    nid,
                    callee.name,
                    None,
                    span,
                )
            # (c) Direct user function call
            if (
                callee_ref is not None
                and callee_ref.kind is BinderKind.function_binding
            ):
                return self._lower_direct_call(call_node, callee_ref, nid, span)

        # Indirect/value call: callee is an arbitrary expression (lambda, let-bound
        # closure, function-value param, etc.).  Named args are rejected by the checker at
        # value-call sites, so only positional args exist here.
        return self._lower_indirect_call(call_node, nid, span)

    def _direct_call_param_types(
        self,
        call_node: Call,
        sig: FunctionSignature,
        binding: tuple[Expr | None, ...],
    ) -> tuple[Type, ...]:
        """Resolve a partial declared-call's parameter types, substituting type vars.

        Reached only from the partial-call lowering path (regular calls read the
        checker's recorded ``function_param_types``), so the call node always has a
        ``FunctionType`` node type and a recorded ``PartialCallSpec``.  Type-var
        bindings come from explicit type arguments when present, otherwise from
        matching hole and non-hole slots against the resulting closure's shape.
        """
        subst: dict[str, Type] = {}
        if call_node.type_args:
            for param_name, type_arg in zip(sig.type_params, call_node.type_args):
                subst[param_name] = self._checked.type_env.resolve_type_expr(type_arg)
        else:
            call_type = self._node_type(call_node.node_id)
            assert isinstance(call_type, FunctionType)
            partial_spec = self._checked.partial_calls.get(call_node.node_id)
            assert partial_spec is not None
            for spec, bound_expr, hole_index in zip(
                sig.params, binding, partial_spec.argument_holes
            ):
                if isinstance(bound_expr, Placeholder):
                    assert hole_index is not None
                    self._match_typevars(spec.type, call_type.params[hole_index], subst)
            self._match_typevars(sig.result, call_type.result, subst)
            for spec, bound_expr in zip(sig.params, binding):
                if bound_expr is not None and not isinstance(bound_expr, Placeholder):
                    self._match_typevars(spec.type, self._node_type(bound_expr.node_id), subst)
        return tuple(substitute(spec.type, subst) for spec in sig.params)

    def _lower_direct_call_with_args(
        self,
        *,
        function_id: FunctionId,
        span: SourceSpan,
        arguments: tuple[IrExpr | UseDefault, ...],
    ) -> IrDirectCall:
        return IrDirectCall(
            location=self._loc(span),
            function_id=function_id,
            arguments=arguments,
        )

    def _lower_direct_call(
        self,
        call_node: "Call",
        callee_ref: BindingRef,
        result_node_id: int,
        span: "SourceSpan",
    ) -> IrDirectCall:
        """Lower a direct call to a named user function."""
        fn_id = self._link.fn_node_to_id.get(callee_ref.decl_node_id)
        assert fn_id is not None, (
            f"compiler bug: no FunctionId for function decl_node_id={callee_ref.decl_node_id!r}"
        )

        # The checker already bound the call; reuse its result (never re-bind).
        binding = self._checked.argument_bindings.function_calls[result_node_id]
        param_types = self._checked.argument_bindings.function_param_types[result_node_id]
        ir_args: list[IrExpr | UseDefault] = []
        for i, (param_type, bound_expr) in enumerate(zip(param_types, binding)):
            if bound_expr is None:
                ir_args.append(UseDefault(param_index=i))
            else:
                ir_args.append(self.lower_coerced(bound_expr, param_type))

        return self._lower_direct_call_with_args(
            function_id=fn_id,
            span=span,
            arguments=tuple(ir_args),
        )

    def _lower_indirect_call_with_args(
        self,
        *,
        callee: IrExpr,
        span: SourceSpan,
        arguments: tuple[IrExpr, ...],
    ) -> IrIndirectCall:
        return IrIndirectCall(
            location=self._loc(span),
            callee=callee,
            arguments=arguments,
        )

    def _lower_indirect_call(
        self,
        call_node: "Call",
        result_node_id: int,
        span: "SourceSpan",
    ) -> IrIndirectCall:
        """Lower an indirect (value) call to IrIndirectCall.

        The callee is an arbitrary expression.  Arguments are positional-only (the
        checker rejects named args at value-call sites).

        Arguments are lowered with ``lower_coerced`` using the callee's FunctionType
        param types.  This preserves the IR invariant that every runtime value matches
        its statically-declared type — without which the static coercion-elision on the
        function body (``lower_coerced(body, return_type)``) is unsound.  The legacy
        interpreter achieves the same observable outcome via a runtime result-coercion in
        ``_apply_closure``; coercing arguments at the call site is equivalent and lets the
        IR remain statically coercion-free inside the closure body.
        """
        callee_ir = self.lower_expr(call_node.callee)
        # Named args are impossible at value-call sites (the checker rejects them).
        assert not call_node.named_args, (
            "compiler bug: named args at indirect call site (checker should have rejected)"
        )
        # Obtain the callee's FunctionType to drive per-arg coercions.
        callee_fn_type = self._node_type(call_node.callee.node_id)
        assert isinstance(callee_fn_type, FunctionType), (
            f"compiler bug: indirect call callee has non-FunctionType node_type"
            f" {callee_fn_type!r}"
        )
        arg_irs: list[IrExpr] = [
            self.lower_coerced(arg, callee_fn_type.params[i])
            for i, arg in enumerate(call_node.args)
        ]
        return self._lower_indirect_call_with_args(
            callee=callee_ir,
            span=span,
            arguments=tuple(arg_irs),
        )

    def _lower_named_constructor_call(
        self,
        result_node_id: int,
        owner_name: str,
        variant: str | None,
        span: "SourceSpan",
    ) -> IrExpr:
        """Lower a constructor call to an IrMake* node.

        Reuses the field→expr binding the checker already computed (never re-binds),
        then builds the Ir node via ``_lower_constructor_from_type``.
        """
        typ = self._checked.node_types.get(result_node_id)
        arg_exprs = self._checked.argument_bindings.constructor_calls[result_node_id]
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
        if typ is None:
            owner_typ = self._checked.type_env.get_type(owner_name)  # pragma: no cover
            if owner_typ is not None:  # pragma: no cover
                return self._lower_constructor_from_type(  # pragma: no cover
                    owner_typ, owner_name, variant, arg_exprs, span
                )
        arg_slots = {
            fname: self.lower_coerced(expr, field_type)
            for fname, field_type in self._constructor_field_types(typ, variant).items()
            if (expr := arg_exprs.get(fname)) is not None
        }
        return self._lower_constructor_from_slots(typ, owner_name, variant, arg_slots, span)

    def _constructor_field_types(
        self,
        typ: Type | None,
        variant: str | None,
    ) -> dict[str, Type]:
        if isinstance(typ, RecordType):
            return dict(self._type_table.record_fields(typ))
        if isinstance(typ, ExceptionType):
            return dict(self._type_table.exception_fields(typ))
        if isinstance(typ, EnumType):
            assert variant is not None, "compiler bug: enum constructor must have variant"
            return dict(self._type_table.enum_variants(typ).get(variant, {}))
        raise AssertionError(  # pragma: no cover
            "compiler bug: constructor field types require a constructor type"
        )

    def _lower_constructor_from_slots(
        self,
        typ: "Type | None",
        owner_name: str,
        variant: str | None,
        arg_slots: "dict[str, IrExpr]",
        span: "SourceSpan",
    ) -> IrExpr:
        """Build the IrMake* node for a constructor from already-lowered slots."""
        loc = self._loc(span)

        if isinstance(typ, RecordType):
            nominal = NominalId(typ.module_id, typ.name)
            # Build fields in declaration order via the shared TypeTable (its
            # TypeDef stores fields as a declaration-ordered tuple).
            ir_fields = tuple(
                (fname, arg_slots[fname]) for fname in self._type_table.record_fields(typ)
            )
            return IrMakeRecord(
                location=loc,
                nominal=nominal,
                display_name=typ.name,
                fields=ir_fields,
            )

        if isinstance(typ, ExceptionType):
            nominal = NominalId(typ.module_id, typ.name)
            # ONE trace id allocation sentinel per construction (auto-fill any
            # declared field not present in arg_slots).
            exc_fields: list[tuple[str, IrExpr | AutoTraceField]] = []
            for fname in self._type_table.exception_fields(typ):
                if fname in arg_slots:
                    exc_fields.append((fname, arg_slots[fname]))
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
            variant_fields = self._type_table.enum_variants(typ).get(variant, {})
            enum_fields = tuple((fname, arg_slots[fname]) for fname in variant_fields)
            return IrMakeEnum(
                location=loc,
                nominal=nominal,
                display_name=typ.name,
                variant=variant,
                fields=enum_fields,
            )

        raise AssertionError(  # pragma: no cover
            f"compiler bug: cannot determine constructor type for {owner_name!r}"
        )

    # ------------------------------------------------------------------
    # Partial call lowering
    # ------------------------------------------------------------------

    def _partial_written_non_holes(self, call_node: Call) -> tuple[Expr, ...]:
        positional = tuple(arg for arg in call_node.args if not isinstance(arg, Placeholder))
        named = tuple(
            named_arg.value
            for named_arg in call_node.named_args
            if not isinstance(named_arg.value, Placeholder)
        )
        return (*positional, *named)

    def _partial_slot_loads(
        self,
        *,
        binding: tuple[Expr | None, ...],
        argument_holes: tuple[int | None, ...],
        target_types: tuple[Type, ...],
        capture_symbols: dict[int, SymbolId],
        param_symbols: tuple[SymbolId, ...],
        span: SourceSpan,
    ) -> tuple[IrExpr | UseDefault, ...]:
        loc = self._loc(span)
        slots: list[IrExpr | UseDefault] = []
        for index, (bound_expr, hole_index, target_type) in enumerate(
            zip(binding, argument_holes, target_types)
        ):
            if bound_expr is None:
                slots.append(UseDefault(param_index=index))
                continue
            if isinstance(bound_expr, Placeholder):
                assert hole_index is not None, "compiler bug: placeholder slot lacks hole index"
                slots.append(IrLoad(location=loc, symbol=param_symbols[hole_index]))
                continue
            sym = capture_symbols[bound_expr.node_id]
            load = IrLoad(location=loc, symbol=sym)
            slots.append(
                self._coerce_ir(load, self._node_type(bound_expr.node_id), target_type, loc)
            )
        return tuple(slots)

    def _partial_declared_body(
        self,
        call_node: Call,
        span: SourceSpan,
        spec: PartialCallSpec,
        capture_symbols: dict[int, SymbolId],
        param_symbols: tuple[SymbolId, ...],
    ) -> IrExpr:
        assert isinstance(call_node.callee, VarRef)
        callee_ref = self._checked.resolved.resolution.get(call_node.callee.node_id)
        assert callee_ref is not None and callee_ref.kind is BinderKind.function_binding
        fn_id = self._link.fn_node_to_id.get(callee_ref.decl_node_id)
        assert fn_id is not None, (
            f"compiler bug: no FunctionId for function decl_node_id={callee_ref.decl_node_id!r}"
        )
        sig = self._checked.type_env.get_function_signature_by_node_id(callee_ref.decl_node_id)
        assert sig is not None, f"compiler bug: no signature for function {callee_ref.name!r}"
        binding = self._checked.argument_bindings.function_calls[call_node.node_id]
        target_types = self._direct_call_param_types(call_node, sig, binding)
        arguments = self._partial_slot_loads(
            binding=binding,
            argument_holes=spec.argument_holes,
            target_types=target_types,
            capture_symbols=capture_symbols,
            param_symbols=param_symbols,
            span=span,
        )
        return self._lower_direct_call_with_args(
            function_id=fn_id,
            span=span,
            arguments=arguments,
        )

    def _partial_value_body(
        self,
        call_node: Call,
        span: SourceSpan,
        spec: PartialCallSpec,
        callee_symbol: SymbolId,
        capture_symbols: dict[int, SymbolId],
        param_symbols: tuple[SymbolId, ...],
    ) -> IrExpr:
        callee_type = self._node_type(call_node.callee.node_id)
        assert isinstance(callee_type, FunctionType)
        binding: tuple[Expr | None, ...] = call_node.args
        value_slots = self._partial_slot_loads(
            binding=binding,
            argument_holes=spec.argument_holes,
            target_types=callee_type.params,
            capture_symbols=capture_symbols,
            param_symbols=param_symbols,
            span=span,
        )
        arguments = tuple(cast(IrExpr, slot) for slot in value_slots)
        return self._lower_indirect_call_with_args(
            callee=IrLoad(location=self._loc(span), symbol=callee_symbol),
            span=span,
            arguments=arguments,
        )

    def _partial_constructor_body(
        self,
        call_node: Call,
        span: SourceSpan,
        spec: PartialCallSpec,
        capture_symbols: dict[int, SymbolId],
        param_symbols: tuple[SymbolId, ...],
    ) -> IrExpr:
        owner_name, variant = self._partial_constructor_owner(call_node)
        partial_type = self._node_type(call_node.node_id)
        assert isinstance(partial_type, FunctionType)
        result_type = partial_type.result
        field_types = self._constructor_field_types(result_type, variant)
        binding_by_name = self._checked.argument_bindings.constructor_calls[call_node.node_id]
        field_order = tuple(binding_by_name)
        binding = tuple(binding_by_name[field_name] for field_name in field_order)
        target_types = tuple(field_types[field_name] for field_name in field_order)
        slots = self._partial_slot_loads(
            binding=binding,
            argument_holes=spec.argument_holes,
            target_types=target_types,
            capture_symbols=capture_symbols,
            param_symbols=param_symbols,
            span=span,
        )
        arg_slots = {
            field_name: cast(IrExpr, slot)
            for field_name, slot in zip(field_order, slots)
        }
        return self._lower_constructor_from_slots(
            result_type,
            owner_name,
            variant,
            arg_slots,
            span,
        )

    def _partial_constructor_owner(self, call_node: Call) -> tuple[str, str | None]:
        callee = call_node.callee
        assert isinstance(callee, VarRef)
        qcr = self._checked.resolved.qualified_constructor_refs.get(callee.node_id)
        if qcr is not None:
            owner_name, variant, _owner_module = qcr
            return owner_name, variant
        cref = self._checked.resolved.constructor_refs.get(callee.node_id)
        if cref is not None:
            return cref.owner_name, cref.variant
        callee_ref = self._checked.resolved.resolution.get(callee.node_id)
        assert callee_ref is not None
        return callee_ref.name, None

    def _lower_partial_call(
        self,
        call_node: Call,
        spec: PartialCallSpec,
        span: SourceSpan,
    ) -> IrExpr:
        partial_type = self._node_type(call_node.node_id)
        assert isinstance(partial_type, FunctionType)
        loc = self._loc(span)
        items: list[IrExpr] = []
        captures: list[IrCapture] = []
        capture_symbols: dict[int, SymbolId] = {}
        callee_symbol: SymbolId | None = None

        if spec.callee_kind == "value":
            callee_symbol = self._alloc_synthetic_sym(mutable=False)
            items.append(
                IrBind(
                    location=loc,
                    symbol=callee_symbol,
                    value=self.lower_expr(call_node.callee),
                )
            )
            captures.append(IrCapture(symbol=callee_symbol, by_cell=False))

        for arg in self._partial_written_non_holes(call_node):
            sym = self._alloc_synthetic_sym(mutable=False)
            capture_symbols[arg.node_id] = sym
            items.append(IrBind(location=loc, symbol=sym, value=self.lower_expr(arg)))
            captures.append(IrCapture(symbol=sym, by_cell=False))

        fn_id = self._alloc_fn()
        fn_sym = self._alloc_synthetic_sym(mutable=False, owner=fn_id)
        params = tuple(
            IrFunctionParam(
                symbol=self._alloc_synthetic_sym(mutable=False, owner=fn_id),
                default=None,
            )
            for _param_type in partial_type.params
        )
        param_symbols = tuple(param.symbol for param in params)

        if spec.callee_kind == "declared":
            body = self._partial_declared_body(
                call_node, span, spec, capture_symbols, param_symbols
            )
        elif spec.callee_kind == "value":
            assert callee_symbol is not None
            body = self._partial_value_body(
                call_node, span, spec, callee_symbol, capture_symbols, param_symbols
            )
        else:
            body = self._partial_constructor_body(
                call_node, span, spec, capture_symbols, param_symbols
            )

        self._link.functions[fn_id] = FunctionDescriptor(
            function_id=fn_id,
            function_symbol=fn_sym,
            module_id=self._module_id,
            params=params,
            body=body,
            param_labels=tuple(repr(param_type) for param_type in partial_type.params),
            result_label=repr(partial_type.result),
        )
        items.append(
            IrMakeClosure(
                location=loc,
                function_id=fn_id,
                captures=tuple(captures),
            )
        )
        return IrBlock(location=loc, items=tuple(items))

    # ------------------------------------------------------------------
    # Block helper
    # ------------------------------------------------------------------

    def _item_is_bottom(self, item: Item) -> bool:
        """Return whether a checked block item unconditionally exits."""
        if isinstance(item, (LetDecl, VarDecl, AssignStmt)):
            return isinstance(self._node_type(item.value.node_id), BottomType)
        return isinstance(self._node_type(item.node_id), BottomType)

    def _lower_block(
        self,
        items: tuple[Item, ...],
        span: SourceSpan,
    ) -> IrBlock:
        """Lower a ``Block``'s items to an ``IrBlock``.

        All items inside a block body are lowered as **nested** (``top_level=False``),
        so any ``let``/``var`` binders they declare are allocated with
        ``public=False`` and do not appear in ``_collect_results``.  Only the
        top-level module-initializer driver passes ``top_level=True``.  Scope
        rejects root-only declarations in nested blocks, so every reachable item
        must lower to a runtime expression.
        """
        real: list[IrExpr] = []
        for item in items:
            ir = self.lower_item(item, top_level=False)
            assert ir is not None, "compiler bug: non-runtime item in nested block"
            real.append(ir)
            if self._item_is_bottom(item):
                break
        assert real, "compiler bug: lowered block has no runtime items"
        return IrBlock(location=self._loc(span), items=tuple(real))

    # ------------------------------------------------------------------
    # Control-flow helpers
    # ------------------------------------------------------------------

    def _lower_if(
        self,
        branches: "tuple[IfBranch, ...]",
        span: "SourceSpan",
        result_type: Type,
    ) -> IrIf:
        """Lower an ``If`` AST node to ``IrIf``."""
        has_else = any(isinstance(br.cond, ElseSentinel) for br in branches)
        ir_branches: list[IrIfBranch] = []
        for branch in branches:
            if isinstance(branch.cond, ElseSentinel):
                ir_branches.append(
                    IrIfBranch(cond=None, body=self.lower_coerced(branch.body, result_type))
                )
            else:
                ir_branches.append(
                    IrIfBranch(
                        cond=self.lower_expr(branch.cond),
                        body=self.lower_coerced(branch.body, result_type),
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
        # Determine nominal + display_name.  ``nominal`` must name the resolved
        # exception's *real* module-qualified identity (built-in, entry, or a
        # library module for a cross-module exception) because specific catches
        # match exactly by ``ExceptionValue.nominal`` at runtime.
        exc_type = clause.exc_type
        if exc_type is None or exc_type == "_" or exc_type == "Exception":
            nominal: NominalId | None = None
            display_name: str | None = None
        else:
            resolved = self._checked.type_env.resolve_named_type(exc_type)
            assert isinstance(resolved, ExceptionType), (
                f"compiler bug: catch clause type {exc_type!r} did not resolve "
                "to an ExceptionType"
            )
            nominal = NominalId(resolved.module_id, resolved.name)
            display_name = resolved.name

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
    # Case expression helpers
    # ------------------------------------------------------------------

    def _lower_case(
        self,
        subject_expr: "Expr",
        branches: "tuple[CaseBranch, ...]",
        span: "SourceSpan",
        result_type: Type,
    ) -> IrCase:
        """Lower a ``Case`` AST node to ``IrCase`` with compiled match plans."""
        ir_arms = tuple(
            IrCaseArm(
                plan=self._compile_plan(branch.pattern),
                body=self.lower_coerced(branch.body, result_type),
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
        makes a missing case a mypy exhaustiveness error.
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

            case ConstructorPattern(name=variant_name):
                field_plans: list[tuple[str, IrMatchPlan]] = [
                    (fname, self._compile_plan(sub_pat))
                    for fname, sub_pat
                    in self._checked.argument_bindings.constructor_patterns[pattern.node_id]
                ]
                return IrConstructorPlan(
                    variant=variant_name,
                    fields=tuple(field_plans),
                )

            case _ as unreachable:  # pragma: no cover
                assert_never(unreachable)

    # ------------------------------------------------------------------
    # Ask/ask-request lowering
    # ------------------------------------------------------------------

    def _lower_ask_call(
        self,
        call_node: "Call",
        span: "SourceSpan",
        *,
        structured_exec: bool,
        is_request: bool = False,
    ) -> IrExpr:
        """Lower an ask() or ask-request() builtin call to IrAsk/IrAskRequest."""
        loc = self._loc(span)
        named_map: dict[str, "NamedArg"] = {na.name: na for na in call_node.named_args}

        # 1. Evaluate the prompt (first positional arg).
        prompt_ir = self.lower_expr(call_node.args[0])

        # 2. Evaluate the agent expression (named arg 'agent:', or default "ask").
        if "agent" in named_map:
            agent_ir: IrExpr = self.lower_expr(named_map["agent"].value)
        else:
            # No agent: named arg → the default agent name "ask" as a text constant.
            # The evaluator will use this TextValue as the agent name.
            agent_ir = IrConstText(location=loc, value="ask")

        # 3. Determine max_attempts from the on_parse_error named arg.
        max_attempts = self._extract_max_attempts(call_node)

        # 4. Build ContractRequest from the checker's contract_spec (if any).
        result_type = self._checked.node_types.get(call_node.node_id)
        is_unit = isinstance(result_type, UnitType)

        spec = self._checked.contract_specs.get(call_node.node_id)
        if is_unit or spec is None:
            # Unit-typed ask: dispatch without output parsing.
            contract_req = ContractRequest(
                codec_name="text",
                strict_json=None,
                json_schema=None,
                decode=None,
                target_type_label="unit" if is_unit else "text",
                structured_exec=structured_exec,
                format_instructions="",
                is_unit=True,
                target_type_kind="unit",
            )
        else:
            json_schema_str, fmt_instr, decode_schema, decode_defs = (
                self._contract_payload_for_spec(call_node.node_id, spec)
            )
            contract_req = ContractRequest(
                codec_name=spec.codec_name,
                strict_json=spec.strict_json,
                json_schema=json_schema_str,
                decode=decode_schema,
                target_type_label=repr(spec.target_type),
                structured_exec=structured_exec,
                format_instructions=fmt_instr,
                is_unit=False,
                target_type_kind=spec.target_type.kind,
                target_type=spec.target_type,
                defs=decode_defs,
            )

        contract_id = self._alloc_contract(contract_req)

        if is_request:
            return IrAskRequest(
                location=loc,
                agent=agent_ir,
                prompt=prompt_ir,
                contract_id=contract_id,
                max_attempts=max_attempts,
            )
        return IrAsk(
            location=loc,
            agent=agent_ir,
            prompt=prompt_ir,
            contract_id=contract_id,
            max_attempts=max_attempts,
        )

    # ------------------------------------------------------------------
    # Exec lowering
    # ------------------------------------------------------------------

    def _lower_exec_call(
        self,
        call_node: "Call",
        span: "SourceSpan",
    ) -> IrExpr:
        """Lower an exec() builtin call to IrExec."""
        loc = self._loc(span)

        # command is first positional arg
        command_ir = self.lower_expr(call_node.args[0])

        max_attempts = self._extract_max_attempts(call_node)

        spec = self._checked.contract_specs.get(call_node.node_id)
        assert spec is not None, "exec always has a contract spec after checking"
        structured_exec = spec.structured_exec
        json_schema_str, fmt_instr, decode_schema, decode_defs = (
            self._contract_payload_for_spec(call_node.node_id, spec)
        )
        contract_req = ContractRequest(
            codec_name=spec.codec_name,
            strict_json=spec.strict_json,
            json_schema=json_schema_str,
            decode=decode_schema,
            target_type_label=repr(spec.target_type),
            structured_exec=structured_exec,
            format_instructions=fmt_instr,
            is_unit=False,
            target_type_kind=spec.target_type.kind,
            target_type=spec.target_type,
            defs=decode_defs,
        )

        contract_id = self._alloc_contract(contract_req)
        return IrExec(
            location=loc,
            command=command_ir,
            contract_id=contract_id,
            max_attempts=max_attempts,
        )

    def _extract_max_attempts(self, call_node: "Call") -> int:
        """Extract max_attempts from the on_parse_error named arg at lowering time."""
        named_map: dict[str, "NamedArg"] = {na.name: na for na in call_node.named_args}
        if "on_parse_error" not in named_map:
            return 1
        policy_expr = named_map["on_parse_error"].value
        if isinstance(policy_expr, Call):
            callee = policy_expr.callee
            if isinstance(callee, VarRef):
                callee_name: str | None = callee.name
            elif isinstance(callee, FieldAccess):
                callee_name = callee.field
            else:
                callee_name = None
            if callee_name == "Retry":
                n_val = next(
                    (arg.value.value for arg in policy_expr.named_args
                     if arg.name == "n" and isinstance(arg.value, IntLit)),
                    0,
                )
                return 1 + n_val
        # Absent or Abort → single attempt
        return 1

    # ------------------------------------------------------------------
    # Item lowering
    # ------------------------------------------------------------------

    def lower_item(self, item: Item, *, top_level: bool = False) -> IrExpr | None:
        """Lower a block item.

        Returns an ``IrExpr`` for nodes with runtime action, or ``None`` for
        purely compile-time declarations (type definitions, function defs, etc.)
        that have no IR representation.

        Module-initializer construction filters out the ``None`` values;
        nested blocks reject root-only declarations before lowering.

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
                binding_type = self._binding_type(nid)
                ir_val = self.lower_coerced(rhs, binding_type)
                return IrBind(location=self._loc(span), symbol=sym, value=ir_val)

            case VarDecl(name=name, value=rhs, span=span, node_id=nid):
                sym = self._alloc_sym(nid, name=name, mutable=True, public=top_level)
                binding_type = self._binding_type(nid)
                ir_val = self.lower_coerced(rhs, binding_type)
                return IrBind(location=self._loc(span), symbol=sym, value=ir_val)

            case AssignStmt(target=target, value=rhs, span=span, node_id=nid):
                return self._lower_assign(target, rhs, span, nid)

            # ----------------------------------------------------------
            # Declarations with no runtime action
            # ----------------------------------------------------------
            case FuncDef() as funcdef:
                if funcdef.is_builtin:
                    return None
                return self._lower_funcdef(funcdef)

            case ParamDecl() as param_decl:
                # Param declarations are only lowered for the entry module.
                # graph.py guards ensure this branch is only called for entry items,
                # so _lower_param_decl is always applicable here.
                self._lower_param_decl(param_decl)
                return None

            case ConfigDecl() as config_decl:
                # Config declarations are entry-only readable bindings; unlike
                # ParamDecl they emit an initializer (IrConfigBind) evaluated in
                # declaration order, NOT hoisted like params.
                return self._lower_config_decl(config_decl)

            case AgentDecl() as agent_decl:
                sym = self._sym_for_decl(agent_decl.node_id)
                loc = self._loc(agent_decl.span)
                return IrBind(
                    location=loc,
                    symbol=sym,
                    value=IrAgentHandle(location=loc, agent_name=agent_decl.name),
                )

            case (
                RecordDef()
                | EnumDef()
                | ExceptionDef()
                | TypeAlias()
                | ProgramDecl()
                | ImportDecl()
                | ExportDecl()
                | InfixDecl()
            ):
                return None

            # ----------------------------------------------------------
            # Expression items — lower as expr
            # ----------------------------------------------------------
            case _:
                # Anything else must be an expression.
                return self.lower_expr(item)

    def _lower_param_decl(self, param: "ParamDecl") -> None:
        """Lower an entry-module ``ParamDecl`` to an ``IrParam`` descriptor.

        Allocates a PUBLIC ``SymbolId`` for the param (owner = entry module) and
        appends an ``IrParam`` to ``self._params``.  Does NOT emit an initializer
        into ``ir_items`` — params are installed by the evaluator's ``run()``
        from ``program.params + param_values`` BEFORE any module initializer runs.
        """
        sym = self._alloc_sym(
            param.node_id,
            name=param.name,
            mutable=False,
            public=True,
            owner=self._module_id,
        )
        binding_type = self._binding_type(param.node_id)
        if param.default is not None:
            default_ir: IrExpr | None = self.lower_coerced(param.default, binding_type)
        else:
            default_ir = None
        ir_param = IrParam(
            symbol=sym,
            public_name=param.name,
            required=(param.default is None),
            default=default_ir,
            location=self._loc(param.span),
            external_decoder=build_param_decoder(binding_type, self._type_table),
        )
        self._params.append(ir_param)

    def _lower_config_decl(self, node: "ConfigDecl") -> IrConfigBind:
        """Lower an entry-module ``ConfigDecl`` to an ``IrConfigBind`` initializer.

        Allocates a PUBLIC ``SymbolId`` for the config key and lowers the source
        value expression (projecting a bare inner-type value into ``some(value)``
        for ``Option[T]`` engine keys).  Unlike ``_lower_param_decl`` this returns
        an initializer node (params are hoisted; config bindings are evaluated in
        declaration order).  Decoding of external CLI/config-file values is handled
        host-side via ``convert_config_value`` — the IR node carries no decoder.
        """
        sym = self._alloc_sym(
            node.node_id,
            name=node.name,
            mutable=False,
            public=True,
            owner=self._module_id,
        )
        declared_type = self._binding_type(node.node_id)
        value_ir: IrExpr | None
        if node.value is not None:
            value_ir = self._lower_config_value(node.value, declared_type)
        else:
            value_ir = None
        return IrConfigBind(
            location=self._loc(node.span),
            symbol=sym,
            public_name=node.name,
            value=value_ir,
        )

    def _lower_config_value(self, expr: Expr, declared_type: Type) -> IrExpr:
        """Lower a config value, projecting a bare ``T`` into ``some(T)`` for Option keys.

        When the engine-key type is ``Option[T]`` and the source value's type is
        not itself an enum (i.e. it is the inner ``T``), wrap the lowered inner
        value in an ``Option::Some`` construction.  Otherwise lower with ordinary
        coercion to the declared type.
        """
        if isinstance(declared_type, EnumType) and declared_type.type_args:
            expr_type = self._node_type(expr.node_id)
            if not isinstance(expr_type, EnumType):
                inner = declared_type.type_args[0]
                inner_ir = self.lower_coerced(expr, inner)
                self._ensure_option_nominal()
                return IrMakeEnum(
                    location=self._loc(expr.span),
                    nominal=NominalId(STD_CORE_ID, "Option"),
                    display_name="Option",
                    variant="Some",
                    fields=(("value", inner_ir),),
                )
        return self.lower_coerced(expr, declared_type)

    def _ensure_option_nominal(self) -> None:
        """Register the ``std.core::Option`` enum nominal if not already present."""
        nominal = NominalId(STD_CORE_ID, "Option")
        if nominal not in self._link.nominals:
            self._link.nominals[nominal] = NominalDescriptor(
                nominal=nominal,
                display_name="Option",
                kind=NominalKind.ENUM,
                fields=(),
                variants=(
                    VariantDescriptor(name="None", fields=()),
                    VariantDescriptor(name="Some", fields=("value",)),
                ),
            )

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
            slot_type = self._binding_type(ref.decl_node_id)
            ir_val = self.lower_coerced(rhs, slot_type)
            return IrAssign(
                location=self._loc(span),
                symbol=sym,
                path=(),
                value=ir_val,
            )

        # IndexTarget: flatten the index path into IrIndexStep list.
        root_type = self._binding_type(ref.decl_node_id)
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
        """Populate ``self._link.nominals`` with all user-declared and built-in nominals.

        Adds:
        - All user-declared record/enum/exception nominals from the entry
          module's type env.
        - All built-in prelude record/enum and exception descriptors keyed by
          NominalId(PRELUDE_ID, name).

        Field, variant, and exception-field names are resolved through the
        shared ``TypeTable`` rather than any embedded map on the handle.
        """
        table = self._type_table
        # User-declared nominals for this lowering unit.
        for name, typ in self._checked.type_env.non_builtin_type_items():
            if isinstance(typ, RecordType):
                nominal = NominalId(typ.module_id, name)
                self._link.nominals[nominal] = NominalDescriptor(
                    nominal=nominal,
                    display_name=name,
                    kind=NominalKind.RECORD,
                    fields=tuple(table.record_fields(typ).keys()),
                    variants=(),
                )
            elif isinstance(typ, EnumType):
                nominal = NominalId(typ.module_id, name)
                variants = tuple(
                    VariantDescriptor(name=vname, fields=tuple(vfields.keys()))
                    for vname, vfields in table.enum_variants(typ).items()
                )
                self._link.nominals[nominal] = NominalDescriptor(
                    nominal=nominal,
                    display_name=name,
                    kind=NominalKind.ENUM,
                    fields=(),
                    variants=variants,
                )
            elif isinstance(typ, ExceptionType):
                nominal = NominalId(typ.module_id, name)
                self._link.nominals[nominal] = NominalDescriptor(
                    nominal=nominal,
                    display_name=name,
                    kind=NominalKind.EXCEPTION,
                    fields=tuple(table.exception_fields(typ).keys()),
                    variants=(),
                )
            else:  # pragma: no cover
                # non_builtin_type_items() only ever yields Record/Enum/Exception
                # handles for a non-generic user declaration (generics live in a
                # separate table, aliases are never registered into _types).
                raise AssertionError(
                    f"compiler bug: non-nominal type {typ!r} for {name!r} in "
                    "non_builtin_type_items()"
                )

        # Generic definitions are stored separately from the ordinary type
        # namespace. Runtime nominal identity erases type arguments, so one
        # descriptor per generic declaration is sufficient for every instance.
        # Field/variant NAMES are read directly off the registered TypeDef
        # template (never instantiated — a generic template has no concrete
        # type_args to substitute).
        for name, generic in self._checked.type_env.all_generic_types().items():
            typ = generic.template
            nominal = NominalId(typ.module_id, name)
            typedef = table.get(typ.module_id, name)
            assert typedef is not None, (
                f"compiler bug: generic type {name!r} has no TypeDef registered"
            )
            if isinstance(typ, RecordType):
                self._link.nominals[nominal] = NominalDescriptor(
                    nominal=nominal,
                    display_name=name,
                    kind=NominalKind.RECORD,
                    fields=tuple(fname for fname, _ in typedef.fields),
                )
            else:
                self._link.nominals[nominal] = NominalDescriptor(
                    nominal=nominal,
                    display_name=name,
                    kind=NominalKind.ENUM,
                    variants=tuple(
                        VariantDescriptor(vname, tuple(fname for fname, _ in vfields))
                        for vname, vfields in typedef.variants
                    ),
                )

        _add_builtin_nominals(self._link.nominals, table)

    def lower(self) -> ExecutableProgram:
        """Lower the checked program to an ``ExecutableProgram``."""
        self._build_nominals()

        body = self._checked.resolved.program.body

        # Phase 1: pre-allocate function symbols and IDs for mutual recursion,
        # and allocate agent symbols so they are resolvable in function bodies.
        for item in body.items:
            if isinstance(item, FuncDef) and not item.is_builtin:
                self._prealloc_funcdef(item)
            elif isinstance(item, AgentDecl):
                self._alloc_sym(
                    item.node_id,
                    name=item.name,
                    mutable=False,
                    public=True,
                    owner=self._module_id,
                )

        # Phase 2: lower all items
        function_initializers: list[IrExpr] = []
        other_initializers: list[IrExpr] = []
        for item in body.items:
            ir = self.lower_item(item, top_level=True)
            if ir is not None:
                target = (
                    function_initializers
                    if isinstance(item, FuncDef) and not item.is_builtin
                    else other_initializers
                )
                target.append(ir)

        entry_mod = ExecutableModule(
            module_id=self._module_id,
            initializers=tuple((*function_initializers, *other_initializers)),
        )

        dry_run_inventory = tuple(
            DryRunEntry(
                callee=csr.callee,
                codec_name=csr.codec_name,
                target_type_label=repr(csr.target_type),
                has_schema=_contract_has_schema(
                    self._checked.contract_specs.get(csr.node_id),
                    self._contract_payloads.get(csr.node_id),
                ),
                parse_policy=csr.parse_policy,
                line=csr.line,
                col=csr.col,
            )
            for csr in self._checked.call_sites
        )
        return ExecutableProgram(
            entry_module=self._module_id,
            modules={self._module_id: entry_mod},
            symbols=dict(self._link.symbols),
            nominals=dict(self._link.nominals),
            sources=dict(self._link.sources),
            functions=dict(self._link.functions),
            params=tuple(self._params),
            contracts=dict(self._link.contracts),
            dry_run_inventory=dry_run_inventory,
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
    contract_payloads: Mapping[int, ContractPayload] | None = None,
) -> ExecutableProgram:
    """Lower a single-module ``CheckedProgram`` to an ``ExecutableProgram``.

    :param checked: the type-checked program to lower.
    :param source_text: the normalised source text (used in the sources table).
    :param source_label: human-readable label for the source (display_name).
    :param validate: when ``True``, run ``validate_ir`` before returning.
    :returns: the linked ``ExecutableProgram`` ready for evaluation.
    :raises NotImplementedError: for unsupported AST nodes.
    :raises AssertionError: for missing checker side-table entries (compiler bugs).
    """
    link = _LinkState()
    source_id = SourceId(link.next_source)
    link.next_source += 1
    normalized = normalize_newlines(source_text)
    link.sources[source_id] = SourceFile(display_name=source_label, normalized_text=normalized)
    lowerer = _Lowerer(
        checked,
        link,
        ENTRY_ID,
        source_id,
        source_text,
        contract_payloads=contract_payloads,
    )
    program = lowerer.lower()
    if validate:
        validate_ir(program)
    return program
