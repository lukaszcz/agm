"""``walk()`` traversal for the AgL syntax package.

``walk(node, callback)`` is a recursive traversal that calls *callback* for
every node in the tree in pre-order (parent before children). It dispatches on
the node's exact class to that class's walker, registered by :func:`_walks`;
every class in :data:`SyntaxNode` has one.

Usage::

    from agm.agl.syntax.visitor import walk

    def show(node: object) -> None:
        if isinstance(node, LetDecl):
            print(f"let {node.name}")

    walk(program, show)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from agm.agl.syntax.nodes import (
    ArrayLit,
    AsPattern,
    AssignStmt,
    Attribute,
    AttributeKeyedArg,
    BinaryOp,
    Block,
    BoolLit,
    Break,
    BuiltinVarDecl,
    Call,
    CallArg,
    Case,
    CaseBranch,
    Cast,
    CatchClause,
    Constraint,
    ConstructorPattern,
    Continue,
    DecimalLit,
    DictEntry,
    DictLit,
    ElseSentinel,
    EnumDef,
    ExceptionDef,
    ExportDecl,
    ExportItem,
    FieldAccess,
    FieldTarget,
    FuncDef,
    If,
    IfBranch,
    ImportDecl,
    ImportItem,
    IndexAccess,
    IndexTarget,
    InfixDecl,
    InterpSegment,
    IntLit,
    IsTest,
    Lambda,
    LetDecl,
    LiteralPattern,
    Loop,
    NamedArg,
    NameTarget,
    NullLit,
    OperatorRef,
    Param,
    PatternField,
    Placeholder,
    Program,
    QualifierChain,
    QualifierSegment,
    Raise,
    RawInfixChain,
    RawInfixOperand,
    RawInfixOperator,
    RawPrefixNot,
    RecordDef,
    RecordUpdate,
    Return,
    ScopeRegion,
    ScopeSegment,
    StringLit,
    Template,
    TextSegment,
    Try,
    TypeAlias,
    TypeApply,
    UnaryNeg,
    UnaryNot,
    UnitLit,
    UseDecl,
    VarDecl,
    VariantDef,
    VariantRef,
    VarPattern,
    VarRef,
    WildcardPattern,
)
from agm.agl.syntax.types import (
    AppliedT,
    ArrayT,
    BoolT,
    DecimalT,
    DictT,
    FuncT,
    IntT,
    JsonT,
    NameT,
    TextT,
    UnitT,
)

# Every node class ``walk`` visits.
type SyntaxNode = (
    Program
    | Attribute
    | AttributeKeyedArg
    | TextT
    | JsonT
    | BoolT
    | IntT
    | DecimalT
    | NameT
    | ArrayT
    | DictT
    | UnitT
    | FuncT
    | AppliedT
    | QualifierSegment
    | QualifierChain
    | ImportItem
    | ImportDecl
    | ExportItem
    | ExportDecl
    | UseDecl
    | ScopeSegment
    | RecordDef
    | VariantDef
    | VariantRef
    | EnumDef
    | ExceptionDef
    | TypeAlias
    | FuncDef
    | Constraint
    | BuiltinVarDecl
    | InfixDecl
    | ScopeRegion
    | LetDecl
    | VarDecl
    | AssignStmt
    | NameTarget
    | IndexTarget
    | FieldTarget
    | UnitLit
    | IntLit
    | DecimalLit
    | BoolLit
    | NullLit
    | StringLit
    | ArrayLit
    | DictEntry
    | DictLit
    | TextSegment
    | InterpSegment
    | Template
    | VarRef
    | FieldAccess
    | IndexAccess
    | NamedArg[CallArg]
    | Placeholder
    | OperatorRef
    | BinaryOp
    | UnaryNot
    | UnaryNeg
    | Cast
    | IsTest
    | TypeApply
    | Call
    | RecordUpdate
    | Param
    | Lambda
    | Block
    | IfBranch
    | If
    | CaseBranch
    | Case
    | Loop
    | CatchClause
    | Try
    | Raise
    | Return
    | RawInfixChain
    | RawInfixOperand
    | RawInfixOperator
    | RawPrefixNot
    | Break
    | Continue
    | WildcardPattern
    | LiteralPattern
    | VarPattern
    | AsPattern
    | PatternField
    | ConstructorPattern
    | ElseSentinel
)

# ---------------------------------------------------------------------------
# walk() — pre-order traversal dispatched on the node's class
# ---------------------------------------------------------------------------

type Callback = Callable[[object], None]
type _Walker = Callable[[SyntaxNode, Callback], None]

_WALKERS: dict[type[object], _Walker] = {}
"""Each :data:`SyntaxNode` class's walker, which walks that node's children in order."""


def walk(node: SyntaxNode, callback: Callback) -> None:
    """Pre-order traversal: call ``callback`` with *node*, then recurse."""
    callback(node)
    _WALKERS[type(node)](node, callback)


def _walks[N: SyntaxNode](cls: type[N], walker: Callable[[N, Callback], None]) -> None:
    """Register *walker* as the walker of *cls* nodes."""
    # ``walk`` hands a walker only nodes of the class it is registered under.
    _WALKERS[cls] = cast("_Walker", walker)


def _walk_leaf(node: SyntaxNode, callback: Callback) -> None:
    """A leaf has no child nodes."""


_WALKERS.update(
    dict.fromkeys(
        (
            TextT,
            JsonT,
            BoolT,
            IntT,
            DecimalT,
            UnitT,
            ScopeSegment,
            Constraint,
            InfixDecl,
            UnitLit,
            IntLit,
            DecimalLit,
            BoolLit,
            NullLit,
            StringLit,
            TextSegment,
            Placeholder,
            OperatorRef,
            RawInfixOperator,
            RawPrefixNot,
            Break,
            Continue,
            WildcardPattern,
            VarPattern,
            ElseSentinel,
        ),
        _walk_leaf,
    )
)


def _walk_attributes(attributes: tuple[Attribute, ...], callback: Callback) -> None:
    """Walk a declaration's attribute prefix before the declaration's own children."""
    for attribute in attributes:
        walk(attribute, callback)


def _walk_program(node: Program, callback: Callback) -> None:
    walk(node.body, callback)


def _walk_attribute(node: Attribute, callback: Callback) -> None:
    for attribute_arg in node.args:
        walk(attribute_arg, callback)
    for attribute_keyed in node.keyed_args:
        walk(attribute_keyed, callback)


def _walk_attribute_keyed_arg(node: AttributeKeyedArg, callback: Callback) -> None:
    walk(node.key, callback)
    walk(node.value, callback)


def _walk_name_t(node: NameT, callback: Callback) -> None:
    if node.qualifier is not None:
        walk(node.qualifier, callback)


def _walk_array_t(node: ArrayT, callback: Callback) -> None:
    walk(node.elem, callback)


def _walk_dict_t(node: DictT, callback: Callback) -> None:
    walk(node.key, callback)
    walk(node.value, callback)


def _walk_func_t(node: FuncT, callback: Callback) -> None:
    for param_t in node.params:
        walk(param_t, callback)
    walk(node.result, callback)


def _walk_applied_t(node: AppliedT, callback: Callback) -> None:
    if node.qualifier is not None:
        walk(node.qualifier, callback)
    for arg in node.args:
        walk(arg, callback)


def _walk_qualifier_segment(node: QualifierSegment, callback: Callback) -> None:
    for type_arg in node.type_args or ():
        walk(type_arg, callback)


def _walk_qualifier_chain(node: QualifierChain, callback: Callback) -> None:
    for segment in node.segments:
        walk(segment, callback)


def _walk_import_item(node: ImportItem, callback: Callback) -> None:
    for scope_segment in node.scope_path:
        walk(scope_segment, callback)


def _walk_import_decl(node: ImportDecl, callback: Callback) -> None:
    for scope_segment in node.scope_path:
        walk(scope_segment, callback)
    for import_item in (*(node.tail or ()), *node.hidden):
        walk(import_item, callback)


def _walk_export_item(node: ExportItem, callback: Callback) -> None:
    for scope_segment in node.scope_path:
        walk(scope_segment, callback)


def _walk_export_decl(node: ExportDecl, callback: Callback) -> None:
    for scope_segment in node.scope_path:
        walk(scope_segment, callback)
    for export_item in (*node.items, *node.hidden):
        walk(export_item, callback)


def _walk_use_decl(node: UseDecl, callback: Callback) -> None:
    for scope_segment in (*node.scope_path, *node.target):
        walk(scope_segment, callback)
    for import_item in (*(node.tail or ()), *node.hidden):
        walk(import_item, callback)


def _walk_record_def(node: RecordDef, callback: Callback) -> None:
    _walk_attributes(node.attributes, callback)
    for scope_segment in node.scope_path:
        walk(scope_segment, callback)
    for f in node.fields:
        walk(f, callback)


def _walk_variant_def(node: VariantDef, callback: Callback) -> None:
    _walk_attributes(node.attributes, callback)
    for f in node.fields:
        walk(f, callback)


def _walk_variant_ref(node: VariantRef, callback: Callback) -> None:
    walk(node.chain, callback)
    for type_arg in node.type_args:
        walk(type_arg, callback)


def _walk_enum_def(node: EnumDef, callback: Callback) -> None:
    _walk_attributes(node.attributes, callback)
    for scope_segment in node.scope_path:
        walk(scope_segment, callback)
    for member in node.members:
        walk(member, callback)


def _walk_exception_def(node: ExceptionDef, callback: Callback) -> None:
    _walk_attributes(node.attributes, callback)
    for scope_segment in node.scope_path:
        walk(scope_segment, callback)
    for f in node.fields:
        walk(f, callback)


def _walk_type_alias(node: TypeAlias, callback: Callback) -> None:
    _walk_attributes(node.attributes, callback)
    for scope_segment in node.scope_path:
        walk(scope_segment, callback)
    walk(node.type_expr, callback)


def _walk_func_def(node: FuncDef, callback: Callback) -> None:
    _walk_attributes(node.attributes, callback)
    for scope_segment in node.scope_path:
        walk(scope_segment, callback)
    for constraint in node.constraints:
        walk(constraint, callback)
    for param in node.params:
        walk(param, callback)
    if node.return_type is not None:
        walk(node.return_type, callback)
    if node.body is not None:
        walk(node.body, callback)


def _walk_builtin_var_decl(node: BuiltinVarDecl, callback: Callback) -> None:
    _walk_attributes(node.attributes, callback)
    walk(node.type_ann, callback)
    if node.default is not None:
        walk(node.default, callback)


def _walk_scope_region(node: ScopeRegion, callback: Callback) -> None:
    walk(node.segment, callback)
    for scope_item in node.items:
        walk(scope_item, callback)


def _walk_let_decl(node: LetDecl, callback: Callback) -> None:
    _walk_attributes(node.attributes, callback)
    if node.type_ann is not None:
        walk(node.type_ann, callback)
    walk(node.value, callback)


def _walk_var_decl(node: VarDecl, callback: Callback) -> None:
    _walk_attributes(node.attributes, callback)
    if node.type_ann is not None:
        walk(node.type_ann, callback)
    walk(node.value, callback)


def _walk_assign_stmt(node: AssignStmt, callback: Callback) -> None:
    walk(node.target, callback)
    walk(node.value, callback)


def _walk_name_target(node: NameTarget, callback: Callback) -> None:
    if node.qualifier is not None:
        walk(node.qualifier, callback)


def _walk_index_target(node: IndexTarget, callback: Callback) -> None:
    walk(node.obj, callback)
    walk(node.index, callback)


def _walk_field_target(node: FieldTarget, callback: Callback) -> None:
    walk(node.obj, callback)


def _walk_array_lit(node: ArrayLit, callback: Callback) -> None:
    for elem in node.elements:
        walk(elem, callback)


def _walk_dict_entry(node: DictEntry, callback: Callback) -> None:
    walk(node.key, callback)
    walk(node.value, callback)


def _walk_dict_lit(node: DictLit, callback: Callback) -> None:
    for entry in node.entries:
        walk(entry, callback)


def _walk_interp_segment(node: InterpSegment, callback: Callback) -> None:
    walk(node.expr, callback)


def _walk_template(node: Template, callback: Callback) -> None:
    for seg in node.segments:
        walk(seg, callback)


def _walk_var_ref(node: VarRef, callback: Callback) -> None:
    if node.qualifier is not None:
        walk(node.qualifier, callback)


def _walk_field_access(node: FieldAccess, callback: Callback) -> None:
    walk(node.obj, callback)


def _walk_index_access(node: IndexAccess, callback: Callback) -> None:
    walk(node.obj, callback)
    walk(node.index, callback)


def _walk_named_arg(node: NamedArg[CallArg], callback: Callback) -> None:
    walk(node.value, callback)


def _walk_binary_op(node: BinaryOp, callback: Callback) -> None:
    walk(node.left, callback)
    walk(node.right, callback)


def _walk_unary_not(node: UnaryNot, callback: Callback) -> None:
    walk(node.operand, callback)


def _walk_unary_neg(node: UnaryNeg, callback: Callback) -> None:
    walk(node.operand, callback)


def _walk_cast(node: Cast, callback: Callback) -> None:
    walk(node.expr, callback)
    walk(node.target_type, callback)


def _walk_is_test(node: IsTest, callback: Callback) -> None:
    walk(node.expr, callback)
    if node.qualifier is not None:
        walk(node.qualifier, callback)


def _walk_type_apply(node: TypeApply, callback: Callback) -> None:
    walk(node.expr, callback)
    for ta in node.type_args:
        walk(ta, callback)


def _walk_call(node: Call, callback: Callback) -> None:
    walk(node.callee, callback)
    for call_arg in node.args:
        walk(call_arg, callback)
    for call_named in node.named_args:
        walk(call_named, callback)
    for ta in node.type_args:
        walk(ta, callback)


def _walk_record_update(node: RecordUpdate, callback: Callback) -> None:
    walk(node.target, callback)
    for update in node.updates:
        walk(update, callback)


def _walk_param(node: Param, callback: Callback) -> None:
    _walk_attributes(node.attributes, callback)
    if node.type_expr is not None:
        walk(node.type_expr, callback)
    if node.default is not None:
        walk(node.default, callback)


def _walk_lambda(node: Lambda, callback: Callback) -> None:
    for param in node.params:
        walk(param, callback)
    if node.return_type is not None:
        walk(node.return_type, callback)
    walk(node.body, callback)


def _walk_block(node: Block, callback: Callback) -> None:
    for block_item in node.items:
        walk(block_item, callback)


def _walk_if_branch(node: IfBranch, callback: Callback) -> None:
    walk(node.cond, callback)
    walk(node.body, callback)


def _walk_if(node: If, callback: Callback) -> None:
    for if_branch in node.branches:
        walk(if_branch, callback)


def _walk_case_branch(node: CaseBranch, callback: Callback) -> None:
    walk(node.pattern, callback)
    walk(node.body, callback)


def _walk_case(node: Case, callback: Callback) -> None:
    walk(node.subject, callback)
    for case_branch in node.branches:
        walk(case_branch, callback)


def _walk_loop(node: Loop, callback: Callback) -> None:
    if node.for_iter is not None:
        walk(node.for_iter, callback)
    if node.for_range_to is not None:
        walk(node.for_range_to, callback)
    if node.for_range_step is not None:
        walk(node.for_range_step, callback)
    if node.while_cond is not None:
        walk(node.while_cond, callback)
    if node.bound is not None:
        walk(node.bound, callback)
    walk(node.body, callback)
    if node.until_cond is not None:
        walk(node.until_cond, callback)


def _walk_catch_clause(node: CatchClause, callback: Callback) -> None:
    walk(node.body, callback)


def _walk_try(node: Try, callback: Callback) -> None:
    walk(node.body, callback)
    for clause in node.handlers:
        walk(clause, callback)


def _walk_raise(node: Raise, callback: Callback) -> None:
    walk(node.exc, callback)


def _walk_return(node: Return, callback: Callback) -> None:
    if node.value is not None:
        walk(node.value, callback)


def _walk_raw_infix_chain(node: RawInfixChain, callback: Callback) -> None:
    for operand in node.operands:
        walk(operand, callback)
    for operator in node.operators:
        walk(operator, callback)


def _walk_raw_infix_operand(node: RawInfixOperand, callback: Callback) -> None:
    walk(node.expr, callback)
    for prefix in node.prefix_nots:
        walk(prefix, callback)


def _walk_literal_pattern(node: LiteralPattern, callback: Callback) -> None:
    walk(node.literal, callback)


def _walk_as_pattern(node: AsPattern, callback: Callback) -> None:
    walk(node.pattern, callback)


def _walk_pattern_field(node: PatternField, callback: Callback) -> None:
    walk(node.pattern, callback)


def _walk_constructor_pattern(node: ConstructorPattern, callback: Callback) -> None:
    if node.qualifier is not None:
        walk(node.qualifier, callback)
    for p in node.positional:
        walk(p, callback)
    for pf in node.named:
        walk(pf, callback)


# Each non-leaf class's walker; leaves share ``_walk_leaf`` above.
_walks(Program, _walk_program)
_walks(Attribute, _walk_attribute)
_walks(AttributeKeyedArg, _walk_attribute_keyed_arg)
_walks(NameT, _walk_name_t)
_walks(ArrayT, _walk_array_t)
_walks(DictT, _walk_dict_t)
_walks(FuncT, _walk_func_t)
_walks(AppliedT, _walk_applied_t)
_walks(QualifierSegment, _walk_qualifier_segment)
_walks(QualifierChain, _walk_qualifier_chain)
_walks(ImportItem, _walk_import_item)
_walks(ImportDecl, _walk_import_decl)
_walks(ExportItem, _walk_export_item)
_walks(ExportDecl, _walk_export_decl)
_walks(UseDecl, _walk_use_decl)
_walks(RecordDef, _walk_record_def)
_walks(VariantDef, _walk_variant_def)
_walks(VariantRef, _walk_variant_ref)
_walks(EnumDef, _walk_enum_def)
_walks(ExceptionDef, _walk_exception_def)
_walks(TypeAlias, _walk_type_alias)
_walks(FuncDef, _walk_func_def)
_walks(BuiltinVarDecl, _walk_builtin_var_decl)
_walks(ScopeRegion, _walk_scope_region)
_walks(LetDecl, _walk_let_decl)
_walks(VarDecl, _walk_var_decl)
_walks(AssignStmt, _walk_assign_stmt)
_walks(NameTarget, _walk_name_target)
_walks(IndexTarget, _walk_index_target)
_walks(FieldTarget, _walk_field_target)
_walks(ArrayLit, _walk_array_lit)
_walks(DictEntry, _walk_dict_entry)
_walks(DictLit, _walk_dict_lit)
_walks(InterpSegment, _walk_interp_segment)
_walks(Template, _walk_template)
_walks(VarRef, _walk_var_ref)
_walks(FieldAccess, _walk_field_access)
_walks(IndexAccess, _walk_index_access)
_walks(NamedArg, _walk_named_arg)
_walks(BinaryOp, _walk_binary_op)
_walks(UnaryNot, _walk_unary_not)
_walks(UnaryNeg, _walk_unary_neg)
_walks(Cast, _walk_cast)
_walks(IsTest, _walk_is_test)
_walks(TypeApply, _walk_type_apply)
_walks(Call, _walk_call)
_walks(RecordUpdate, _walk_record_update)
_walks(Param, _walk_param)
_walks(Lambda, _walk_lambda)
_walks(Block, _walk_block)
_walks(IfBranch, _walk_if_branch)
_walks(If, _walk_if)
_walks(CaseBranch, _walk_case_branch)
_walks(Case, _walk_case)
_walks(Loop, _walk_loop)
_walks(CatchClause, _walk_catch_clause)
_walks(Try, _walk_try)
_walks(Raise, _walk_raise)
_walks(Return, _walk_return)
_walks(RawInfixChain, _walk_raw_infix_chain)
_walks(RawInfixOperand, _walk_raw_infix_operand)
_walks(LiteralPattern, _walk_literal_pattern)
_walks(AsPattern, _walk_as_pattern)
_walks(PatternField, _walk_pattern_field)
_walks(ConstructorPattern, _walk_constructor_pattern)
