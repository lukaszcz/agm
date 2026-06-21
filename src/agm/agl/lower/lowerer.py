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

from typing import TYPE_CHECKING, assert_never

if TYPE_CHECKING:
    from agm.agl.syntax.nodes import AssignTarget

from agm.agl.ir.ids import Location, SourceId, SymbolId
from agm.agl.ir.nodes import (
    IrAssign,
    IrBind,
    IrBlock,
    IrCoerce,
    IrConstBool,
    IrConstDecimal,
    IrConstInt,
    IrConstJsonNull,
    IrConstText,
    IrConstUnit,
    IrExpr,
    IrLoad,
    IrMakeDict,
    IrMakeList,
)
from agm.agl.ir.program import (
    ExecutableModule,
    ExecutableProgram,
    SourceFile,
    SymbolDescriptor,
)
from agm.agl.ir.validate import validate_ir
from agm.agl.lower.coercions import compile_coercion
from agm.agl.modules.ids import ENTRY_ID
from agm.agl.syntax.nodes import (
    AgentDecl,
    AssignStmt,
    BinaryOp,
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
    IntLit,
    IsTest,
    Item,
    Lambda,
    LetDecl,
    ListLit,
    NameTarget,
    NullLit,
    ParamDecl,
    ProgramDecl,
    Raise,
    RecordDef,
    StringLit,
    Template,
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
from agm.agl.typecheck.types import DictType, ListType, Type

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
            # Variable reference → IrLoad
            # ----------------------------------------------------------
            case VarRef(node_id=nid, span=span):
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
            # Unsupported M2 nodes — will be M3/M4
            # ----------------------------------------------------------
            case (
                BinaryOp()
                | UnaryNot()
                | UnaryNeg()
                | Call()
                | Lambda()
                | If()
                | Case()
                | Do()
                | Try()
                | Raise()
                | FieldAccess()
                | IndexAccess()
                | Cast()
                | IsTest()
                | Template()
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
        """Lower a simple-name assignment (M2: IndexTarget not supported)."""
        if not isinstance(target, NameTarget):
            raise NotImplementedError(
                "Indexed-assignment paths are not yet supported in M2 lowering "
                f"(got target type {type(target).__name__!r})"
            )
        ref = self._checked.resolved.resolution.get(assign_node_id)
        assert ref is not None, (
            f"compiler bug: no resolution for AssignStmt node_id={assign_node_id!r}"
        )
        sym = self._sym_for_decl(ref.decl_node_id)
        slot_type = self._binding_type_for(ref.decl_node_id)
        ir_val = self.lower_coerced(rhs, slot_type)
        return IrAssign(
            location=self._loc(span),
            symbol=sym,
            path=(),
            value=ir_val,
        )

    # ------------------------------------------------------------------
    # Top-level entry point
    # ------------------------------------------------------------------

    def lower(self) -> ExecutableProgram:
        """Lower the checked program to an ``ExecutableProgram``."""
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
            nominals={},
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
