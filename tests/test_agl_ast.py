"""Tests for the AgL AST package (agm.agl.syntax).

Covers:
- SourceSpan construction and import from diagnostics
- TypeExpr hierarchy (including UnitT, AgentT, FuncT)
- All AST node types: Program/Block, declarations (FuncDef), binders, expressions,
  patterns
- Current nodes: UnitLit, Call, Param, FuncDef, Lambda, Block, If/IfBranch,
  Case/CaseBranch, Loop, Try/CatchClause
- Removed nodes are truly absent (AgentCall, PassStmt, PrintStmt, ExprStmt,
  DoUntil, IfStmt, CaseStmt, CaseExpr, IfExpr, TryCatch, CallOptions,
  AbortPolicy, RetryPolicy)
- Equality semantics: equal structure with different spans/node_ids compare equal
- Immutability: frozen dataclasses reject mutation
- walk traversal visits every node kind
- Tuple-typed children (not lists)
- ELSE sentinel type for IfBranch.cond
- BinaryOp with closed operator set
- DecimalLit holds decimal.Decimal
- Union membership for Expr, Binder, Declaration, Item, Pattern, TemplateSegment
"""

from __future__ import annotations

import decimal
import importlib
from dataclasses import FrozenInstanceError, fields

import pytest

from agm.agl.syntax import (
    # sentinel
    ELSE,
    AppliedT,
    ArrayLit,
    ArrayT,
    AsPattern,
    AssignStmt,
    AssignTarget,
    Attribute,
    BinaryOp,
    Binder,
    BinOp,
    Block,
    BoolLit,
    BoolT,
    Break,
    BuiltinVarDecl,
    Call,
    Case,
    CaseBranch,
    Cast,
    CatchClause,
    ConstructorPattern,
    Continue,
    DecimalLit,
    DecimalT,
    Declaration,
    DictEntry,
    DictLit,
    DictT,
    ElseSentinel,
    EnumDef,
    ExceptionDef,
    Expr,
    FieldAccess,
    FieldTarget,
    FuncDef,
    FuncT,
    If,
    IfBranch,
    IndexAccess,
    IndexTarget,
    InterpSegment,
    IntLit,
    IntT,
    IsTest,
    Item,
    JsonT,
    Lambda,
    LetDecl,
    LiteralPattern,
    Loop,
    NamedArg,
    NameT,
    NameTarget,
    NullLit,
    Param,
    Pattern,
    PatternField,
    Placeholder,
    Program,
    QualifierAnchor,
    QualifierChain,
    Raise,
    RecordDef,
    Return,
    # spans
    ScopeRegion,
    ScopeSegment,
    SourceSpan,
    StringLit,
    Template,
    TemplateSegment,
    TextSegment,
    # types
    TextT,
    Try,
    TypeAlias,
    TypeApply,
    UnaryNeg,
    UnaryNot,
    UnitLit,
    UnitT,
    VarDecl,
    VariantDef,
    VariantRef,
    VarPattern,
    # nodes – expressions
    VarRef,
    # nodes – patterns
    WildcardPattern,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def span(sl: int = 1, sc: int = 0, el: int = 1, ec: int = 1) -> SourceSpan:
    return SourceSpan(
        start_line=sl,
        start_col=sc,
        end_line=el,
        end_col=ec,
        start_offset=0,
        end_offset=1,
    )


def nid(n: int = 1) -> int:
    return n


# ---------------------------------------------------------------------------
# SourceSpan
# ---------------------------------------------------------------------------


class TestSourceSpan:
    def test_construction_positional_fields(self) -> None:
        s = SourceSpan(
            start_line=1,
            start_col=0,
            end_line=1,
            end_col=10,
            start_offset=0,
            end_offset=10,
        )
        assert s.start_line == 1
        assert s.start_col == 0
        assert s.end_line == 1
        assert s.end_col == 10
        assert s.start_offset == 0
        assert s.end_offset == 10

    def test_equality(self) -> None:
        s1 = SourceSpan(1, 0, 1, 10, 0, 10)
        s2 = SourceSpan(1, 0, 1, 10, 0, 10)
        assert s1 == s2

    def test_frozen(self) -> None:
        s = SourceSpan(1, 0, 1, 5, 0, 5)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(s, "start_line", 99)

    def test_diagnostics_import_same_class(self) -> None:
        # The lexer imports SourceSpan from diagnostics; it must be the same class.
        # Import lazily to avoid triggering the full agm.agl pipeline (which is
        # temporarily broken while downstream stages are being updated).
        from agm.agl.syntax.spans import SourceSpan as SpansSourceSpan

        assert SourceSpan is SpansSourceSpan

    def test_offsets_are_required(self) -> None:
        # Offsets are mandatory: a 4-arg construction is an argument error.
        args = (1, 1, 1, 10)
        with pytest.raises(TypeError):
            SourceSpan(*args)


# ---------------------------------------------------------------------------
# Nullary nodes
# ---------------------------------------------------------------------------

# Every AST node kind whose whole content is its position: a type expression, a
# literal, a jump, or a pattern.  They share one contract, so they are checked
# together rather than one construction at a time.
_NULLARY_NODES = [
    TextT,
    JsonT,
    BoolT,
    IntT,
    DecimalT,
    UnitT,
    UnitLit,
    NullLit,
    Break,
    Continue,
    WildcardPattern,
]


@pytest.mark.parametrize("node_type", _NULLARY_NODES, ids=lambda t: t.__name__)
def test_nullary_node_equality_ignores_span_and_node_id(node_type: type) -> None:
    """Two nullary nodes of a kind are the same node wherever they were written."""
    first = node_type(span=span(1, 0, 1, 5), node_id=1)
    second = node_type(span=span(9, 2, 9, 7), node_id=99)

    assert first == second


@pytest.mark.parametrize("node_type", _NULLARY_NODES, ids=lambda t: t.__name__)
def test_nullary_node_is_frozen(node_type: type) -> None:
    """No pass may retag a node in place; artifacts wrap the AST instead."""
    node = node_type(span=span(), node_id=1)

    with pytest.raises((FrozenInstanceError, AttributeError)):
        setattr(node, "span", span(2, 0, 2, 4))


# ---------------------------------------------------------------------------
# TypeExpr hierarchy
# ---------------------------------------------------------------------------


class TestTypeExprs:
    def _s(self) -> SourceSpan:
        return span()

    def test_name_t(self) -> None:
        t = NameT(name="MyType", span=self._s(), node_id=1)
        assert t.name == "MyType"

    def test_array_t(self) -> None:
        elem = TextT(span=self._s(), node_id=2)
        t = ArrayT(elem=elem, span=self._s(), node_id=1)
        assert t.elem is elem

    def test_dict_t(self) -> None:
        val = IntT(span=self._s(), node_id=2)
        t = DictT(value=val, span=self._s(), node_id=1)
        assert t.value is val

    def test_func_t_no_params(self) -> None:
        result = IntT(span=self._s(), node_id=2)
        t = FuncT(params=(), result=result, span=self._s(), node_id=1)
        assert t.params == ()
        assert t.result is result

    def test_func_t_with_params(self) -> None:
        p1 = IntT(span=self._s(), node_id=2)
        p2 = TextT(span=self._s(), node_id=3)
        result = BoolT(span=self._s(), node_id=4)
        t = FuncT(params=(p1, p2), result=result, span=self._s(), node_id=1)
        assert isinstance(t.params, tuple)
        assert len(t.params) == 2
        assert t.params[0] is p1
        assert t.params[1] is p2

    def test_name_t_equality(self) -> None:
        t1 = NameT(name="Foo", span=span(1, 0, 1, 3), node_id=1)
        t2 = NameT(name="Foo", span=span(2, 0, 2, 3), node_id=99)
        assert t1 == t2

    def test_name_t_inequality(self) -> None:
        t1 = NameT(name="Foo", span=span(), node_id=1)
        t2 = NameT(name="Bar", span=span(), node_id=1)
        assert t1 != t2

    def test_array_t_equality(self) -> None:
        a = ArrayT(elem=TextT(span=span(), node_id=2), span=span(), node_id=1)
        b = ArrayT(elem=TextT(span=span(3, 0, 3, 4), node_id=99), span=span(5, 0, 5, 4), node_id=50)
        assert a == b

    def test_func_t_equality_ignores_span_node_id(self) -> None:
        r1 = IntT(span=span(1, 0, 1, 3), node_id=10)
        r2 = IntT(span=span(5, 0, 5, 3), node_id=20)
        t1 = FuncT(params=(), result=r1, span=span(1, 0, 1, 10), node_id=1)
        t2 = FuncT(params=(), result=r2, span=span(9, 0, 9, 10), node_id=99)
        assert t1 == t2

    def test_unit_t_equality(self) -> None:
        t1 = UnitT(span=span(1, 0, 1, 4), node_id=1)
        t2 = UnitT(span=span(5, 0, 5, 4), node_id=50)
        assert t1 == t2

    def test_type_frozen(self) -> None:
        t = TextT(span=span(), node_id=1)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(t, "span", span())


# ---------------------------------------------------------------------------
# Literal nodes
# ---------------------------------------------------------------------------


class TestLiterals:
    def _s(self) -> SourceSpan:
        return span()

    def test_int_lit(self) -> None:
        node = IntLit(value=42, span=self._s(), node_id=1)
        assert node.value == 42

    def test_decimal_lit(self) -> None:
        d = decimal.Decimal("3.14")
        node = DecimalLit(value=d, span=self._s(), node_id=1)
        assert node.value == d

    def test_bool_lit_true(self) -> None:
        node = BoolLit(value=True, span=self._s(), node_id=1)
        assert node.value is True

    def test_bool_lit_false(self) -> None:
        node = BoolLit(value=False, span=self._s(), node_id=1)
        assert node.value is False

    def test_string_lit(self) -> None:
        node = StringLit(value="hello", span=self._s(), node_id=1)
        assert node.value == "hello"

    def test_array_lit(self) -> None:
        elems: tuple[Expr, ...] = (IntLit(value=1, span=self._s(), node_id=2),)
        node = ArrayLit(elements=elems, span=self._s(), node_id=1)
        assert isinstance(node.elements, tuple)
        assert len(node.elements) == 1

    def test_array_lit_empty(self) -> None:
        node = ArrayLit(elements=(), span=self._s(), node_id=1)
        assert node.elements == ()

    def test_dict_lit(self) -> None:
        entry = DictEntry(
            key=StringLit(value="k", span=self._s(), node_id=3),
            value=IntLit(value=1, span=self._s(), node_id=4),
            span=self._s(),
            node_id=2,
        )
        node = DictLit(entries=((entry,)), span=self._s(), node_id=1)
        assert isinstance(node.entries, tuple)

    def test_equality_ignores_span_node_id(self) -> None:
        a = IntLit(value=5, span=span(1, 0, 1, 1), node_id=1)
        b = IntLit(value=5, span=span(9, 3, 9, 4), node_id=999)
        assert a == b

    def test_inequality_different_value(self) -> None:
        a = IntLit(value=1, span=span(), node_id=1)
        b = IntLit(value=2, span=span(), node_id=1)
        assert a != b


# ---------------------------------------------------------------------------
# Template nodes
# ---------------------------------------------------------------------------


class TestTemplate:
    def _s(self) -> SourceSpan:
        return span()

    def test_text_segment(self) -> None:
        seg = TextSegment(text="hello", span=self._s(), node_id=1)
        assert seg.text == "hello"

    def test_interp_segment(self) -> None:
        expr = VarRef(name="x", span=self._s(), node_id=2)
        seg = InterpSegment(expr=expr, span=self._s(), node_id=1)
        assert seg.expr is expr

    def test_template_segments_are_tuple(self) -> None:
        segs: tuple[TemplateSegment, ...] = (
            TextSegment(text="hello ", span=self._s(), node_id=2),
            InterpSegment(
                expr=VarRef(name="name", span=self._s(), node_id=3),
                span=self._s(),
                node_id=4,
            ),
        )
        tmpl = Template(segments=segs, span=self._s(), node_id=1)
        assert isinstance(tmpl.segments, tuple)
        assert len(tmpl.segments) == 2


# ---------------------------------------------------------------------------
# Expression nodes
# ---------------------------------------------------------------------------


class TestExpressions:
    def _s(self) -> SourceSpan:
        return span()

    def test_var_ref(self) -> None:
        node = VarRef(name="x", span=self._s(), node_id=1)
        assert node.name == "x"

    def test_field_access(self) -> None:
        obj = VarRef(name="obj", span=self._s(), node_id=2)
        node = FieldAccess(obj=obj, field="attr", span=self._s(), node_id=1)
        assert node.field == "attr"
        assert node.obj is obj

    def test_index_access(self) -> None:
        obj = VarRef(name="xs", span=self._s(), node_id=2)
        index = IntLit(value=0, span=self._s(), node_id=3)
        node = IndexAccess(obj=obj, index=index, span=self._s(), node_id=1)
        assert node.obj is obj
        assert node.index is index

    def test_binary_op(self) -> None:
        left = IntLit(value=1, span=self._s(), node_id=2)
        right = IntLit(value=2, span=self._s(), node_id=3)
        node = BinaryOp(op=BinOp.ADD, left=left, right=right, span=self._s(), node_id=1)
        assert node.op is BinOp.ADD

    def test_all_binary_ops_exist(self) -> None:
        ops = {
            BinOp.EQ,
            BinOp.NEQ,
            BinOp.LT,
            BinOp.LE,
            BinOp.GT,
            BinOp.GE,
            BinOp.IN,
            BinOp.AND,
            BinOp.OR,
            BinOp.ADD,
            BinOp.SUB,
            BinOp.MUL,
            BinOp.DIV,
        }
        assert ops == set(BinOp)

    def test_unary_not(self) -> None:
        operand = BoolLit(value=True, span=self._s(), node_id=2)
        node = UnaryNot(operand=operand, span=self._s(), node_id=1)
        assert node.operand is operand

    def test_unary_neg(self) -> None:
        operand = IntLit(value=5, span=self._s(), node_id=2)
        node = UnaryNeg(operand=operand, span=self._s(), node_id=1)
        assert node.operand is operand

    def test_is_test(self) -> None:
        expr = VarRef(name="x", span=self._s(), node_id=2)
        node = IsTest(
            expr=expr, qualifier=None, variant="Some", negated=False, span=self._s(), node_id=1
        )
        assert node.negated is False
        assert node.variant == "Some"

    def test_is_test_negated(self) -> None:
        expr = VarRef(name="x", span=self._s(), node_id=2)
        node = IsTest(
            expr=expr, qualifier=None, variant="None_", negated=True, span=self._s(), node_id=1
        )
        assert node.negated is True

    def test_placeholder(self) -> None:
        bare = Placeholder(index=None, span=self._s(), node_id=1)
        numbered = Placeholder(index=2, span=self._s(), node_id=2)
        assert bare.index is None
        assert numbered.index == 2

    def test_equality_ignores_span_node_id(self) -> None:
        a = VarRef(name="x", span=span(1, 0, 1, 1), node_id=10)
        b = VarRef(name="x", span=span(5, 3, 5, 4), node_id=99)
        assert a == b

    def test_inequality_different_name(self) -> None:
        a = VarRef(name="x", span=span(), node_id=1)
        b = VarRef(name="y", span=span(), node_id=1)
        assert a != b


# ---------------------------------------------------------------------------
# Current expression nodes
# ---------------------------------------------------------------------------


class TestCallNode:
    def _s(self) -> SourceSpan:
        return span()

    def test_call_no_args(self) -> None:
        callee = VarRef(name="f", span=self._s(), node_id=2)
        node = Call(callee=callee, args=(), named_args=(), span=self._s(), node_id=1)
        assert node.callee is callee
        assert node.args == ()
        assert node.named_args == ()

    def test_call_positional_args(self) -> None:
        callee = VarRef(name="f", span=self._s(), node_id=2)
        a1 = IntLit(value=1, span=self._s(), node_id=3)
        a2 = IntLit(value=2, span=self._s(), node_id=4)
        node = Call(callee=callee, args=(a1, a2), named_args=(), span=self._s(), node_id=1)
        assert isinstance(node.args, tuple)
        assert len(node.args) == 2
        assert node.args[0] is a1

    def test_call_named_args(self) -> None:
        callee = VarRef(name="ask", span=self._s(), node_id=2)
        prompt = StringLit(value="hi", span=self._s(), node_id=3)
        agent_ref = VarRef(name="reviewer", span=self._s(), node_id=4)
        named = NamedArg(name="agent", value=agent_ref, span=self._s(), node_id=5)
        node = Call(
            callee=callee,
            args=(prompt,),
            named_args=(named,),
            span=self._s(),
            node_id=1,
        )
        assert isinstance(node.named_args, tuple)
        assert node.named_args[0].name == "agent"

    def test_call_equality_ignores_span_node_id(self) -> None:
        callee = VarRef(name="f", span=span(), node_id=5)
        a = Call(callee=callee, args=(), named_args=(), span=span(1, 0, 1, 3), node_id=1)
        b = Call(callee=callee, args=(), named_args=(), span=span(9, 0, 9, 3), node_id=99)
        assert a == b

    def test_call_frozen(self) -> None:
        callee = VarRef(name="f", span=span(), node_id=2)
        node = Call(callee=callee, args=(), named_args=(), span=span(), node_id=1)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(node, "callee", callee)


class TestParamNode:
    def _s(self) -> SourceSpan:
        return span()

    def test_param_required(self) -> None:
        t = IntT(span=self._s(), node_id=2)
        p = Param(name="x", type_expr=t, default=None, span=self._s(), node_id=1)
        assert p.name == "x"
        assert p.type_expr is t
        assert p.default is None
        assert p.mutable is False

    def test_param_with_default(self) -> None:
        t = IntT(span=self._s(), node_id=2)
        default = IntLit(value=0, span=self._s(), node_id=3)
        p = Param(
            name="n",
            type_expr=t,
            default=default,
            span=self._s(),
            node_id=1,
        )
        assert p.default is default

    def test_param_equality_ignores_span_node_id(self) -> None:
        t1 = IntT(span=span(1, 0, 1, 3), node_id=10)
        t2 = IntT(span=span(5, 0, 5, 3), node_id=20)
        a = Param(
            name="x",
            type_expr=t1,
            default=None,
            span=span(1, 0, 1, 5),
            node_id=1,
        )
        b = Param(
            name="x",
            type_expr=t2,
            default=None,
            span=span(9, 0, 9, 5),
            node_id=99,
        )
        assert a == b

    def test_param_frozen(self) -> None:
        t = IntT(span=span(), node_id=2)
        p = Param(name="x", type_expr=t, default=None, span=span(), node_id=1)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(p, "name", "y")


class TestFuncDefNode:
    def _s(self) -> SourceSpan:
        return span()

    def test_func_def_minimal(self) -> None:
        ret = IntT(span=self._s(), node_id=3)
        body = VarRef(name="x", span=self._s(), node_id=4)
        fd = FuncDef(
            name="identity",
            params=(),
            return_type=ret,
            body=body,
            span=self._s(),
            node_id=1,
        )
        assert fd.name == "identity"
        assert fd.params == ()
        assert fd.return_type is ret
        assert fd.body is body

    def test_func_def_with_params(self) -> None:
        p = Param(
            name="n",
            type_expr=IntT(span=self._s(), node_id=3),
            default=None,
            span=self._s(),
            node_id=2,
        )
        ret = IntT(span=self._s(), node_id=4)
        body = VarRef(name="n", span=self._s(), node_id=5)
        fd = FuncDef(
            name="f",
            params=(p,),
            return_type=ret,
            body=body,
            span=self._s(),
            node_id=1,
        )
        assert isinstance(fd.params, tuple)
        assert fd.params[0] is p

    def test_func_def_equality_ignores_span_node_id(self) -> None:
        ret = IntT(span=span(1, 0, 1, 3), node_id=10)
        body = IntLit(value=1, span=span(1, 0, 1, 1), node_id=11)
        a = FuncDef(
            name="f",
            params=(),
            return_type=ret,
            body=body,
            span=span(1, 0, 1, 20),
            node_id=1,
        )
        b = FuncDef(
            name="f",
            params=(),
            return_type=ret,
            body=body,
            span=span(9, 0, 9, 20),
            node_id=99,
        )
        assert a == b

    def test_func_def_frozen(self) -> None:
        ret = IntT(span=span(), node_id=2)
        body = IntLit(value=0, span=span(), node_id=3)
        fd = FuncDef(name="f", params=(), return_type=ret, body=body, span=span(), node_id=1)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(fd, "name", "g")


class TestLambdaNode:
    def _s(self) -> SourceSpan:
        return span()

    def test_lambda_no_return_type(self) -> None:
        p = Param(
            name="x",
            type_expr=IntT(span=self._s(), node_id=2),
            default=None,
            span=self._s(),
            node_id=3,
        )
        body = VarRef(name="x", span=self._s(), node_id=4)
        lam = Lambda(params=(p,), return_type=None, body=body, span=self._s(), node_id=1)
        assert lam.return_type is None
        assert lam.body is body

    def test_lambda_with_return_type(self) -> None:
        p = Param(
            name="x",
            type_expr=IntT(span=self._s(), node_id=2),
            default=None,
            span=self._s(),
            node_id=3,
        )
        ret = IntT(span=self._s(), node_id=4)
        body = VarRef(name="x", span=self._s(), node_id=5)
        lam = Lambda(params=(p,), return_type=ret, body=body, span=self._s(), node_id=1)
        assert lam.return_type is ret

    def test_lambda_params_is_tuple(self) -> None:
        body = UnitLit(span=self._s(), node_id=2)
        lam = Lambda(params=(), return_type=None, body=body, span=self._s(), node_id=1)
        assert isinstance(lam.params, tuple)

    def test_lambda_equality_ignores_span_node_id(self) -> None:
        body = IntLit(value=1, span=span(1, 0, 1, 1), node_id=5)
        a = Lambda(params=(), return_type=None, body=body, span=span(1, 0, 1, 10), node_id=1)
        b = Lambda(params=(), return_type=None, body=body, span=span(9, 0, 9, 10), node_id=99)
        assert a == b

    def test_lambda_frozen(self) -> None:
        body = UnitLit(span=span(), node_id=2)
        lam = Lambda(params=(), return_type=None, body=body, span=span(), node_id=1)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(lam, "body", body)


class TestBlockNode:
    def _s(self) -> SourceSpan:
        return span()

    def test_block_empty(self) -> None:
        blk = Block(items=(), span=self._s(), node_id=1)
        assert blk.items == ()

    def test_block_with_expr(self) -> None:
        item = IntLit(value=42, span=self._s(), node_id=2)
        blk = Block(items=(item,), span=self._s(), node_id=1)
        assert isinstance(blk.items, tuple)
        assert len(blk.items) == 1

    def test_block_with_binder_and_expr(self) -> None:
        val = IntLit(value=1, span=self._s(), node_id=3)
        let = LetDecl(
            pattern=VarPattern(name="x", span=self._s(), node_id=5),
            type_ann=None,
            value=val,
            span=self._s(),
            node_id=2,
        )
        result = VarRef(name="x", span=self._s(), node_id=4)
        blk = Block(items=(let, result), span=self._s(), node_id=1)
        assert len(blk.items) == 2

    def test_block_equality_ignores_span_node_id(self) -> None:
        item = IntLit(value=1, span=span(1, 0, 1, 1), node_id=5)
        a = Block(items=(item,), span=span(1, 0, 1, 5), node_id=1)
        b = Block(items=(item,), span=span(9, 0, 9, 5), node_id=99)
        assert a == b

    def test_block_frozen(self) -> None:
        blk = Block(items=(), span=span(), node_id=1)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(blk, "items", ())


class TestIfNode:
    def _s(self) -> SourceSpan:
        return span()

    def test_if_single_branch(self) -> None:
        cond = BoolLit(value=True, span=self._s(), node_id=2)
        body = IntLit(value=1, span=self._s(), node_id=3)
        branch = IfBranch(cond=cond, body=body, span=self._s(), node_id=4)
        node = If(branches=(branch,), span=self._s(), node_id=1)
        assert isinstance(node.branches, tuple)
        assert len(node.branches) == 1

    def test_if_branch_with_else(self) -> None:
        cond = BoolLit(value=True, span=self._s(), node_id=2)
        t_body = IntLit(value=1, span=self._s(), node_id=3)
        f_body = IntLit(value=0, span=self._s(), node_id=4)
        branch_if = IfBranch(cond=cond, body=t_body, span=self._s(), node_id=5)
        branch_else = IfBranch(cond=ELSE, body=f_body, span=self._s(), node_id=6)
        node = If(branches=(branch_if, branch_else), span=self._s(), node_id=1)
        assert node.branches[1].cond is ELSE
        assert isinstance(node.branches[1].cond, ElseSentinel)

    def test_if_branch_cond_is_expr(self) -> None:
        cond = BinaryOp(
            op=BinOp.GT,
            left=VarRef(name="x", span=self._s(), node_id=3),
            right=IntLit(value=0, span=self._s(), node_id=4),
            span=self._s(),
            node_id=2,
        )
        body = StringLit(value="pos", span=self._s(), node_id=5)
        branch = IfBranch(cond=cond, body=body, span=self._s(), node_id=1)
        assert branch.cond is cond

    def test_if_branch_body_is_single_expr(self) -> None:
        # In AgL, branch body is a single Expr (not a tuple).
        cond = BoolLit(value=True, span=self._s(), node_id=2)
        body = UnitLit(span=self._s(), node_id=3)
        branch = IfBranch(cond=cond, body=body, span=self._s(), node_id=1)
        # body is an Expr, not a tuple
        assert not isinstance(branch.body, tuple)

    def test_if_equality_ignores_span_node_id(self) -> None:
        cond = BoolLit(value=True, span=span(1, 0, 1, 4), node_id=2)
        body = UnitLit(span=span(1, 0, 1, 2), node_id=3)
        branch = IfBranch(cond=cond, body=body, span=span(1, 0, 1, 10), node_id=4)
        a = If(branches=(branch,), span=span(1, 0, 1, 20), node_id=1)
        b = If(branches=(branch,), span=span(9, 0, 9, 20), node_id=99)
        assert a == b


class TestCaseNode:
    def _s(self) -> SourceSpan:
        return span()

    def test_case_expr(self) -> None:
        subject = VarRef(name="x", span=self._s(), node_id=2)
        pat = WildcardPattern(span=self._s(), node_id=4)
        body = NullLit(span=self._s(), node_id=5)
        branch = CaseBranch(pattern=pat, body=body, span=self._s(), node_id=3)
        node = Case(subject=subject, branches=(branch,), span=self._s(), node_id=1)
        assert isinstance(node.branches, tuple)
        assert node.branches[0].pattern is pat

    def test_case_branch_body_is_single_expr(self) -> None:
        pat = WildcardPattern(span=self._s(), node_id=2)
        body = IntLit(value=0, span=self._s(), node_id=3)
        branch = CaseBranch(pattern=pat, body=body, span=self._s(), node_id=1)
        assert not isinstance(branch.body, tuple)

    def test_case_equality_ignores_span_node_id(self) -> None:
        subject = VarRef(name="x", span=span(1, 0, 1, 1), node_id=5)
        pat = WildcardPattern(span=span(1, 0, 1, 1), node_id=6)
        body = UnitLit(span=span(1, 0, 1, 2), node_id=7)
        branch = CaseBranch(pattern=pat, body=body, span=span(1, 0, 1, 10), node_id=8)
        a = Case(subject=subject, branches=(branch,), span=span(1, 0, 1, 20), node_id=1)
        b = Case(subject=subject, branches=(branch,), span=span(9, 0, 9, 20), node_id=99)
        assert a == b


class TestLoopNode:
    def _s(self) -> SourceSpan:
        return span()

    def test_loop_no_bound(self) -> None:
        body = UnitLit(span=self._s(), node_id=2)
        node = Loop(
            for_var=None,
            for_iter=None,
            for_range_to=None,
            for_range_down=False,
            for_range_step=None,
            while_cond=None,
            bound=None,
            body=body,
            until_cond=None,
            span=self._s(),
            node_id=1,
        )
        assert node.bound is None
        assert node.body is body
        assert node.until_cond is None

    def test_loop_with_bound(self) -> None:
        body = UnitLit(span=self._s(), node_id=2)
        cond = BoolLit(value=False, span=self._s(), node_id=3)
        bound = IntLit(value=10, span=self._s(), node_id=4)
        node = Loop(
            for_var=None,
            for_iter=None,
            for_range_to=None,
            for_range_down=False,
            for_range_step=None,
            while_cond=None,
            bound=bound,
            body=body,
            until_cond=cond,
            span=self._s(),
            node_id=1,
        )
        assert node.bound is bound

    def test_loop_with_until_cond(self) -> None:
        # body is a single Expr (not a tuple of stmts)
        body = Block(items=(UnitLit(span=self._s(), node_id=3),), span=self._s(), node_id=2)
        cond = BoolLit(value=True, span=self._s(), node_id=4)
        node = Loop(
            for_var=None,
            for_iter=None,
            for_range_to=None,
            for_range_down=False,
            for_range_step=None,
            while_cond=None,
            bound=None,
            body=body,
            until_cond=cond,
            span=self._s(),
            node_id=1,
        )
        assert isinstance(node.body, Block)
        assert node.until_cond is cond

    def test_loop_equality_ignores_span_node_id(self) -> None:
        body = UnitLit(span=span(1, 0, 1, 2), node_id=5)
        cond = BoolLit(value=True, span=span(1, 0, 1, 4), node_id=6)
        bound = IntLit(value=5, span=span(1, 0, 1, 1), node_id=7)
        a = Loop(
            for_var=None,
            for_iter=None,
            for_range_to=None,
            for_range_down=False,
            for_range_step=None,
            while_cond=None,
            bound=bound,
            body=body,
            until_cond=cond,
            span=span(1, 0, 1, 20),
            node_id=1,
        )
        b = Loop(
            for_var=None,
            for_iter=None,
            for_range_to=None,
            for_range_down=False,
            for_range_step=None,
            while_cond=None,
            bound=bound,
            body=body,
            until_cond=cond,
            span=span(9, 0, 9, 20),
            node_id=99,
        )
        assert a == b

    def test_loop_frozen(self) -> None:
        body = UnitLit(span=span(), node_id=2)
        node = Loop(
            for_var=None,
            for_iter=None,
            for_range_to=None,
            for_range_down=False,
            for_range_step=None,
            while_cond=None,
            bound=None,
            body=body,
            until_cond=None,
            span=span(),
            node_id=1,
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(node, "bound", 5)


class TestBreakContinueNodes:
    """Tests for the Break and Continue AST nodes (leaf nodes with BottomType)."""

    def _s(self) -> SourceSpan:
        return span()

    def test_walk_visits_break_as_leaf(self) -> None:
        from agm.agl.syntax.visitor import walk

        node = Break(span=self._s(), node_id=1)
        visited: list[object] = []
        walk(node, visited.append)
        assert visited == [node]  # only the Break itself — no children

    def test_walk_visits_continue_as_leaf(self) -> None:
        from agm.agl.syntax.visitor import walk

        node = Continue(span=self._s(), node_id=1)
        visited: list[object] = []
        walk(node, visited.append)
        assert visited == [node]  # only the Continue itself — no children


class TestTryNode:
    def _s(self) -> SourceSpan:
        return span()

    def test_try_no_handlers(self) -> None:
        body = IntLit(value=1, span=self._s(), node_id=2)
        node = Try(body=body, handlers=(), span=self._s(), node_id=1)
        assert node.body is body
        assert node.handlers == ()

    def test_catch_clause_catch_all(self) -> None:
        body = UnitLit(span=self._s(), node_id=2)
        clause = CatchClause(exc_type=None, binding=None, body=body, span=self._s(), node_id=1)
        assert clause.exc_type is None
        assert clause.binding is None
        assert clause.body is body

    def test_catch_clause_typed_binding(self) -> None:
        body = UnitLit(span=self._s(), node_id=2)
        clause = CatchClause(
            exc_type="MyError",
            binding="e",
            body=body,
            span=self._s(),
            node_id=1,
        )
        assert clause.exc_type == "MyError"
        assert clause.binding == "e"

    def test_catch_clause_body_is_expr(self) -> None:
        # In AgL, body is a single Expr, not a tuple of stmts.
        body = UnitLit(span=self._s(), node_id=2)
        clause = CatchClause(exc_type=None, binding=None, body=body, span=self._s(), node_id=1)
        assert not isinstance(clause.body, tuple)

    def test_try_with_handler(self) -> None:
        body = IntLit(value=1, span=self._s(), node_id=2)
        handler_body = IntLit(value=0, span=self._s(), node_id=3)
        clause = CatchClause(
            exc_type="Error", binding="e", body=handler_body, span=self._s(), node_id=4
        )
        node = Try(body=body, handlers=(clause,), span=self._s(), node_id=1)
        assert isinstance(node.handlers, tuple)
        assert len(node.handlers) == 1

    def test_try_equality_ignores_span_node_id(self) -> None:
        body = IntLit(value=1, span=span(1, 0, 1, 1), node_id=5)
        a = Try(body=body, handlers=(), span=span(1, 0, 1, 10), node_id=1)
        b = Try(body=body, handlers=(), span=span(9, 0, 9, 10), node_id=99)
        assert a == b


# ---------------------------------------------------------------------------
# Binder nodes
# ---------------------------------------------------------------------------


class TestBinders:
    def _s(self) -> SourceSpan:
        return span()

    def test_let_decl_has_one_pattern_field(self) -> None:
        val = IntLit(value=1, span=self._s(), node_id=3)
        pattern = VarPattern(name="x", span=self._s(), node_id=2)
        node = LetDecl(pattern=pattern, type_ann=None, value=val, span=self._s(), node_id=1)
        assert node.pattern is pattern
        assert node.type_ann is None
        assert tuple(field.name for field in fields(LetDecl)) == (
            "pattern",
            "type_ann",
            "value",
            "span",
            "node_id",
            "scope_path",
            "attributes",
        )

    def test_let_decl_with_pattern_and_type_is_frozen_and_structurally_equal(self) -> None:
        val = IntLit(value=1, span=self._s(), node_id=3)
        pattern = ConstructorPattern(
            qualifier=None,
            name="Point",
            positional=(VarPattern(name="x", span=self._s(), node_id=4),),
            named=(),
            span=self._s(),
            node_id=2,
        )
        t = IntT(span=self._s(), node_id=5)
        node = LetDecl(pattern=pattern, type_ann=t, value=val, span=self._s(), node_id=1)
        equal = LetDecl(
            pattern=pattern,
            type_ann=t,
            value=val,
            span=span(2, 0, 2, 20),
            node_id=99,
        )
        assert node == equal
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(node, "pattern", pattern)

    def test_var_decl(self) -> None:
        val = IntLit(value=0, span=self._s(), node_id=2)
        node = VarDecl(name="count", type_ann=None, value=val, span=self._s(), node_id=1)
        assert node.name == "count"

    def test_let_and_var_decl_scope_path_defaults_to_empty(self) -> None:
        val = IntLit(value=0, span=self._s(), node_id=2)
        pattern = VarPattern(name="x", span=self._s(), node_id=3)
        let_node = LetDecl(pattern=pattern, type_ann=None, value=val, span=self._s(), node_id=1)
        var_node = VarDecl(name="x", type_ann=None, value=val, span=self._s(), node_id=1)
        assert let_node.scope_path == ()
        assert var_node.scope_path == ()

    def test_let_and_var_decl_scope_path_participates_in_equality(self) -> None:
        val = IntLit(value=0, span=self._s(), node_id=2)
        pattern = VarPattern(name="x", span=self._s(), node_id=3)
        segment = ScopeSegment(name="A", span=self._s(), node_id=4)
        plain_let = LetDecl(pattern=pattern, type_ann=None, value=val, span=self._s(), node_id=1)
        scoped_let = LetDecl(
            pattern=pattern,
            type_ann=None,
            value=val,
            span=self._s(),
            node_id=1,
            scope_path=(segment,),
        )
        plain_var = VarDecl(name="x", type_ann=None, value=val, span=self._s(), node_id=1)
        scoped_var = VarDecl(
            name="x", type_ann=None, value=val, span=self._s(), node_id=1, scope_path=(segment,)
        )
        assert plain_let != scoped_let
        assert plain_var != scoped_var

    def test_assign_stmt(self) -> None:
        val = IntLit(value=5, span=self._s(), node_id=2)
        target = NameTarget(name="count", span=self._s(), node_id=3)
        node = AssignStmt(target=target, value=val, span=self._s(), node_id=1)
        assert node.target is target

    def test_name_target(self) -> None:
        node = NameTarget(name="count", span=self._s(), node_id=1)
        assert node.name == "count"

    def test_index_target(self) -> None:
        obj = VarRef(name="xs", span=self._s(), node_id=2)
        index = IntLit(value=0, span=self._s(), node_id=3)
        node = IndexTarget(obj=obj, index=index, span=self._s(), node_id=1)
        assert node.obj is obj
        assert node.index is index

    def test_field_target(self) -> None:
        obj = VarRef(name="record", span=self._s(), node_id=2)
        node = FieldTarget(obj=obj, field="value", span=self._s(), node_id=1)
        assert node.obj is obj
        assert node.field == "value"

    def test_raise_is_expr(self) -> None:
        # Raise is in the Expr union (bottom type).
        expr = VarRef(name="err", span=self._s(), node_id=2)
        node = Raise(exc=expr, span=self._s(), node_id=1)
        assert node.exc is expr

    def test_binder_equality_ignores_span_node_id(self) -> None:
        val = IntLit(value=1, span=span(1, 0, 1, 1), node_id=5)
        pattern = VarPattern(name="x", span=span(), node_id=6)
        a = LetDecl(pattern=pattern, type_ann=None, value=val, span=span(1, 0, 1, 10), node_id=1)
        b = LetDecl(pattern=pattern, type_ann=None, value=val, span=span(9, 0, 9, 10), node_id=99)
        assert a == b

    def test_binder_frozen(self) -> None:
        val = IntLit(value=1, span=span(), node_id=2)
        pattern = VarPattern(name="x", span=span(), node_id=3)
        node = LetDecl(pattern=pattern, type_ann=None, value=val, span=span(), node_id=1)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(node, "pattern", pattern)


# ---------------------------------------------------------------------------
# Declaration nodes
# ---------------------------------------------------------------------------


class TestDeclarations:
    def _s(self) -> SourceSpan:
        return span()

    def test_field_as_param(self) -> None:
        t = TextT(span=self._s(), node_id=2)
        f = Param(
            name="title",
            type_expr=t,
            default=None,
            span=self._s(),
            node_id=1,
        )
        assert f.name == "title"
        assert f.type_expr is t
        assert (f.mutable, f.attributes) == (False, ())

    def test_record_def(self) -> None:
        t = IntT(span=self._s(), node_id=3)
        f = Param(
            name="x",
            type_expr=t,
            default=None,
            span=self._s(),
            node_id=2,
        )
        node = RecordDef(name="Point", fields=(f,), span=self._s(), node_id=1)
        assert node.name == "Point"
        assert isinstance(node.fields, tuple)

    def test_variant_def(self) -> None:
        t = IntT(span=self._s(), node_id=3)
        f = Param(
            name="val",
            type_expr=t,
            default=None,
            span=self._s(),
            node_id=2,
        )
        v = VariantDef(name="Some", fields=(f,), span=self._s(), node_id=1)
        assert v.name == "Some"

    def test_enum_def(self) -> None:
        variant_a = VariantDef(name="A", fields=(), span=self._s(), node_id=2)
        variant_b = VariantDef(name="B", fields=(), span=self._s(), node_id=3)
        node = EnumDef(name="Color", members=(variant_a, variant_b), span=self._s(), node_id=1)
        assert isinstance(node.members, tuple)
        assert len(node.members) == 2

    def test_variant_ref(self) -> None:
        chain = QualifierChain(
            anchor=QualifierAnchor.CURRENT_MODULE,
            segments=(),
            member="Shared",
            span=self._s(),
            node_id=2,
        )
        ref = VariantRef(chain=chain, span=self._s(), node_id=1)
        assert ref.chain is chain

    def test_type_alias(self) -> None:
        t = ArrayT(elem=TextT(span=self._s(), node_id=3), span=self._s(), node_id=2)
        node = TypeAlias(name="Names", type_expr=t, span=self._s(), node_id=1)
        assert node.name == "Names"

    def test_func_def_is_declaration(self) -> None:
        import typing

        args = typing.get_args(Declaration)
        assert FuncDef in args

    def test_exception_def_fields(self) -> None:
        """ExceptionDef stores name, fields, base, and the builtin flag."""
        t = TextT(span=self._s(), node_id=3)
        f = Param(
            name="msg",
            type_expr=t,
            default=None,
            span=self._s(),
            node_id=2,
        )
        node = ExceptionDef(name="MyErr", fields=(f,), base=None, span=self._s(), node_id=1)
        assert node.name == "MyErr"
        assert isinstance(node.fields, tuple)
        assert len(node.fields) == 1
        assert node.fields[0].name == "msg"
        assert node.base is None
        assert node.is_builtin is False

    def test_exception_def_with_base(self) -> None:
        node = ExceptionDef(name="DerivedErr", fields=(), base="BaseErr", span=self._s(), node_id=1)
        assert node.base == "BaseErr"

    def test_exception_def_equality_ignores_span(self) -> None:
        t = IntT(span=self._s(), node_id=3)
        f = Param(
            name="code",
            type_expr=t,
            default=None,
            span=self._s(),
            node_id=2,
        )
        a = ExceptionDef(name="E", fields=(f,), base=None, span=span(1, 0, 1, 1), node_id=1)
        b = ExceptionDef(name="E", fields=(f,), base=None, span=span(9, 0, 9, 1), node_id=99)
        assert a == b

    def test_exception_def_is_declaration(self) -> None:
        """ExceptionDef is part of the Declaration union."""
        import typing

        args = typing.get_args(Declaration)
        assert ExceptionDef in args

    def test_exception_def_walk_visits_fields(self) -> None:
        """walk(ExceptionDef) visits the node and each Param in its fields."""
        from agm.agl.syntax.visitor import walk

        s = self._s()
        t = IntT(span=s, node_id=3)
        field = Param(name="code", type_expr=t, default=None, span=s, node_id=2)
        exc = ExceptionDef(name="MyErr", fields=(field,), base=None, span=s, node_id=1)

        visited: list[object] = []
        walk(exc, visited.append)

        assert exc in visited
        assert field in visited
        assert t in visited


# ---------------------------------------------------------------------------
# Pattern nodes
# ---------------------------------------------------------------------------


class TestPatterns:
    def _s(self) -> SourceSpan:
        return span()

    def test_literal_pattern(self) -> None:
        lit = IntLit(value=42, span=self._s(), node_id=2)
        p = LiteralPattern(literal=lit, span=self._s(), node_id=1)
        assert p.literal is lit

    def test_var_pattern(self) -> None:
        p = VarPattern(name="x", span=self._s(), node_id=1)
        assert p.name == "x"

    def test_as_pattern_is_frozen_and_wraps_an_inner_pattern(self) -> None:
        inner = VarPattern(name="value", span=self._s(), node_id=2)
        pattern = AsPattern(pattern=inner, name="whole", span=self._s(), node_id=1)
        assert pattern.pattern is inner
        assert pattern.name == "whole"
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(pattern, "name", "other")

    def test_constructor_pattern_no_fields(self) -> None:
        p = ConstructorPattern(
            qualifier=None, name="None_", positional=(), named=(), span=self._s(), node_id=1
        )
        assert p.name == "None_"
        assert p.positional == ()
        assert p.named == ()

    def test_constructor_pattern_with_fields(self) -> None:
        pf = PatternField(
            name="val",
            pattern=VarPattern(name="v", span=self._s(), node_id=3),
            span=self._s(),
            node_id=2,
        )
        p = ConstructorPattern(
            qualifier=None,
            name="Some",
            positional=(),
            named=(pf,),
            span=self._s(),
            node_id=1,
        )
        assert isinstance(p.named, tuple)
        assert p.named[0].name == "val"

    def test_pattern_equality_ignores_span_node_id(self) -> None:
        a = WildcardPattern(span=span(1, 0, 1, 1), node_id=1)
        b = WildcardPattern(span=span(5, 0, 5, 1), node_id=50)
        assert a == b


# ---------------------------------------------------------------------------
# Program node
# ---------------------------------------------------------------------------


class TestProgram:
    def _s(self) -> SourceSpan:
        return span()

    def test_empty_program(self) -> None:
        blk = Block(items=(), span=self._s(), node_id=2)
        prog = Program(body=blk, span=self._s(), node_id=0)
        assert prog.body is blk
        assert prog.body.items == ()

    def test_program_with_items(self) -> None:
        item = IntLit(value=42, span=self._s(), node_id=3)
        blk = Block(items=(item,), span=self._s(), node_id=2)
        prog = Program(body=blk, span=self._s(), node_id=1)
        assert isinstance(prog.body, Block)
        assert len(prog.body.items) == 1

    def test_program_equality_ignores_span_node_id(self) -> None:
        item = IntLit(value=1, span=span(1, 0, 1, 1), node_id=5)
        blk = Block(items=(item,), span=span(1, 0, 1, 5), node_id=2)
        a = Program(body=blk, span=span(1, 0, 1, 5), node_id=1)
        b = Program(body=blk, span=span(9, 0, 9, 5), node_id=99)
        assert a == b

    def test_program_body_is_block(self) -> None:
        blk = Block(items=(), span=self._s(), node_id=2)
        prog = Program(body=blk, span=self._s(), node_id=1)
        assert isinstance(prog.body, Block)

    def test_program_frozen(self) -> None:
        blk = Block(items=(), span=span(), node_id=2)
        prog = Program(body=blk, span=span(), node_id=1)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(prog, "body", blk)


# ---------------------------------------------------------------------------
# walk
# ---------------------------------------------------------------------------


class TestWalk:
    """Verify that walk() visits every node kind in a tree."""

    def _s(self) -> SourceSpan:
        return span()

    def _build_tree(self) -> Program:
        """Build a Program whose Block exercises every node kind.

        Every type, expression, binder, declaration, and pattern node that
        exists in the AST must appear somewhere in the returned tree so that
        the walk-coverage tests can confirm they are all reachable.
        """
        s = self._s()

        text_t = TextT(span=s, node_id=100)
        int_t = IntT(span=s, node_id=101)
        bool_t = BoolT(span=s, node_id=102)
        json_t = JsonT(span=s, node_id=103)
        decimal_t = DecimalT(span=s, node_id=104)
        name_t = NameT(name="MyT", span=s, node_id=105)
        list_t = ArrayT(elem=text_t, span=s, node_id=106)
        dict_t = DictT(value=int_t, span=s, node_id=107)
        unit_t = UnitT(span=s, node_id=108)
        func_t = FuncT(params=(int_t, text_t), result=bool_t, span=s, node_id=111)
        applied_t = AppliedT(name="Pair", args=(int_t, text_t), span=s, node_id=112)

        # Declarations exercise the current type and declaration nodes.
        field_int = Param(name="x", type_expr=int_t, default=None, span=s, node_id=200)
        field_bool = Param(name="flag", type_expr=bool_t, default=None, span=s, node_id=201)
        field_json = Param(name="data", type_expr=json_t, default=None, span=s, node_id=202)
        field_decimal = Param(name="price", type_expr=decimal_t, default=None, span=s, node_id=203)
        field_name = Param(name="ref", type_expr=name_t, default=None, span=s, node_id=204)
        field_dict = Param(name="meta", type_expr=dict_t, default=None, span=s, node_id=205)
        record_def = RecordDef(
            name="Point",
            fields=(field_int, field_bool, field_json, field_decimal, field_name, field_dict),
            span=s,
            node_id=210,
        )

        variant_field = Param(name="val", type_expr=text_t, default=None, span=s, node_id=211)
        variant_def = VariantDef(name="Some", fields=(variant_field,), span=s, node_id=212)
        variant_ref = VariantRef(
            chain=QualifierChain(
                anchor=QualifierAnchor.CURRENT_MODULE,
                segments=(),
                member="Shared",
                span=s,
                node_id=213,
            ),
            span=s,
            node_id=214,
        )
        enum_def = EnumDef(name="Opt", members=(variant_def, variant_ref), span=s, node_id=215)

        type_alias = TypeAlias(name="Names", type_expr=list_t, span=s, node_id=214)
        exc_field = Param(
            name="msg",
            type_expr=text_t,
            default=None,
            span=s,
            node_id=2140,
        )
        exception_def = ExceptionDef(
            name="MyErr",
            fields=(exc_field,),
            base=None,
            span=s,
            node_id=2141,
        )
        p_unit = Param(name="u", type_expr=unit_t, default=None, span=s, node_id=218)
        p_receiver = Param(
            name="self",
            type_expr=None,
            default=None,
            span=s,
            node_id=220,
        )
        p_func = Param(
            name="f",
            type_expr=func_t,
            default=IntLit(value=0, span=s, node_id=220),
            span=s,
            node_id=221,
        )
        func_def = FuncDef(
            name="helper",
            params=(p_unit, p_receiver, p_func),
            return_type=unit_t,
            body=UnitLit(span=s, node_id=222),
            span=s,
            node_id=223,
        )

        # --- Literals ---
        int_lit = IntLit(value=1, span=s, node_id=300)
        dec_lit = DecimalLit(value=decimal.Decimal("1.5"), span=s, node_id=301)
        bool_lit = BoolLit(value=True, span=s, node_id=302)
        null_lit = NullLit(span=s, node_id=303)
        str_lit = StringLit(value="hello", span=s, node_id=304)
        unit_lit = UnitLit(span=s, node_id=305)

        dict_entry = DictEntry(key=str_lit, value=int_lit, span=s, node_id=306)
        dict_lit = DictLit(entries=(dict_entry,), span=s, node_id=307)
        list_lit = ArrayLit(elements=(int_lit,), span=s, node_id=308)

        # --- Template ---
        text_seg = TextSegment(text="hi ", span=s, node_id=309)
        var_ref_tmpl = VarRef(name="name", span=s, node_id=310)
        interp_seg = InterpSegment(expr=var_ref_tmpl, span=s, node_id=311)
        template = Template(segments=(text_seg, interp_seg), span=s, node_id=312)

        # --- Expressions ---
        var_ref = VarRef(name="x", span=s, node_id=400)
        field_access = FieldAccess(obj=var_ref, field="y", span=s, node_id=401)
        index_access = IndexAccess(obj=var_ref, index=int_lit, span=s, node_id=417)

        binary_op = BinaryOp(op=BinOp.ADD, left=int_lit, right=int_lit, span=s, node_id=404)
        unary_not = UnaryNot(operand=bool_lit, span=s, node_id=405)
        unary_neg = UnaryNeg(operand=int_lit, span=s, node_id=406)
        is_test = IsTest(
            expr=var_ref, qualifier=None, variant="Some", negated=False, span=s, node_id=407
        )

        # Call node: paren call with positional + named args
        ask_ref = VarRef(name="ask", span=s, node_id=408)
        reviewer_ref = VarRef(name="reviewer", span=s, node_id=409)
        named_agent = NamedArg(name="agent", value=reviewer_ref, span=s, node_id=410)
        placeholder = Placeholder(index=None, span=s, node_id=4100)
        call_node = Call(
            callee=ask_ref,
            args=(template, placeholder),
            named_args=(named_agent,),
            span=s,
            node_id=411,
        )

        # Lambda with return type
        lam_param = Param(
            name="x",
            type_expr=int_t,
            default=None,
            span=s,
            node_id=412,
        )
        lam = Lambda(
            params=(lam_param,),
            return_type=int_t,
            body=VarRef(name="x", span=s, node_id=413),
            span=s,
            node_id=414,
        )

        # Lambda without return type (to exercise None branch in walk)
        lam_no_ret = Lambda(
            params=(),
            return_type=None,
            body=unit_lit,
            span=s,
            node_id=415,
        )

        # --- Patterns ---
        wildcard_pat = WildcardPattern(span=s, node_id=500)
        lit_pat = LiteralPattern(literal=int_lit, span=s, node_id=501)
        var_pat = VarPattern(name="v", span=s, node_id=502)
        pat_field = PatternField(name="val", pattern=var_pat, span=s, node_id=503)
        ctor_pat = ConstructorPattern(
            qualifier=None, name="Some", positional=(), named=(pat_field,), span=s, node_id=504
        )
        as_pat = AsPattern(pattern=ctor_pat, name="whole", span=s, node_id=5040)

        # Case with multiple patterns
        case_branch_wildcard = CaseBranch(pattern=wildcard_pat, body=null_lit, span=s, node_id=505)
        case_branch_lit = CaseBranch(pattern=lit_pat, body=unit_lit, span=s, node_id=506)
        case_branch_ctor = CaseBranch(pattern=as_pat, body=bool_lit, span=s, node_id=507)
        case_node = Case(
            subject=var_ref,
            branches=(case_branch_wildcard, case_branch_lit, case_branch_ctor),
            span=s,
            node_id=508,
        )

        # If with condition branch and else branch
        if_branch_cond = IfBranch(cond=bool_lit, body=int_lit, span=s, node_id=509)
        if_branch_else = IfBranch(cond=ELSE, body=null_lit, span=s, node_id=510)
        if_node = If(branches=(if_branch_cond, if_branch_else), span=s, node_id=511)

        # Loop node
        do_body = Block(items=(unit_lit,), span=s, node_id=512)
        do_limit = IntLit(value=5, span=s, node_id=5120)
        do_node = Loop(
            for_var=None,
            for_iter=None,
            for_range_to=None,
            for_range_down=False,
            for_range_step=None,
            while_cond=None,
            bound=do_limit,
            body=do_body,
            until_cond=bool_lit,
            span=s,
            node_id=513,
        )

        # Try/catch
        catch_body = unit_lit
        catch_clause = CatchClause(
            exc_type="MyError", binding="e", body=catch_body, span=s, node_id=514
        )
        try_node = Try(body=int_lit, handlers=(catch_clause,), span=s, node_id=515)

        # Raise
        raise_node = Raise(exc=var_ref, span=s, node_id=516)

        # Break / Continue (leaf nodes — no children)
        break_node = Break(span=s, node_id=517)
        continue_node = Continue(span=s, node_id=518)

        # --- Binders ---
        let_decl = LetDecl(
            pattern=VarPattern(name="a", span=s, node_id=6000),
            type_ann=None,
            value=int_lit,
            span=s,
            node_id=600,
        )
        let_with_type = LetDecl(
            pattern=VarPattern(name="c", span=s, node_id=6010),
            type_ann=bool_t,
            value=bool_lit,
            span=s,
            node_id=601,
        )
        let_applied = LetDecl(
            pattern=VarPattern(name="p", span=s, node_id=6080),
            type_ann=applied_t,
            value=null_lit,
            span=s,
            node_id=608,
        )
        var_decl = VarDecl(name="b", type_ann=None, value=str_lit, span=s, node_id=602)
        var_with_type = VarDecl(name="d", type_ann=json_t, value=null_lit, span=s, node_id=603)
        name_target = NameTarget(name="b", span=s, node_id=604)
        index_target = IndexTarget(obj=var_ref, index=int_lit, span=s, node_id=605)
        field_target = FieldTarget(obj=var_ref, field="value", span=s, node_id=606)
        assign_stmt = AssignStmt(target=name_target, value=index_access, span=s, node_id=606)
        indexed_assign_stmt = AssignStmt(target=index_target, value=int_lit, span=s, node_id=607)

        # Block at the top level
        top_block = Block(
            items=(
                record_def,
                enum_def,
                exception_def,
                type_alias,
                func_def,
                let_decl,
                let_with_type,
                let_applied,
                var_decl,
                var_with_type,
                assign_stmt,
                indexed_assign_stmt,
                AssignStmt(target=field_target, value=int_lit, span=s, node_id=609),
                # expressions directly in block
                var_ref,
                field_access,
                index_access,
                binary_op,
                unary_not,
                unary_neg,
                is_test,
                call_node,
                lam,
                lam_no_ret,
                case_node,
                if_node,
                do_node,
                try_node,
                raise_node,
                break_node,
                continue_node,
                dec_lit,
                dict_lit,
                list_lit,
            ),
            span=s,
            node_id=1,
        )

        return Program(body=top_block, span=s, node_id=0)

    def test_walk_visits_program(self) -> None:
        from agm.agl.syntax.visitor import walk

        prog = self._build_tree()
        visited: list[object] = []
        walk(prog, visited.append)
        assert any(isinstance(n, Program) for n in visited)

    def test_walk_visits_all_binder_kinds(self) -> None:
        from agm.agl.syntax.visitor import walk

        prog = self._build_tree()
        visited: list[object] = []
        walk(prog, visited.append)
        kinds = {type(n) for n in visited}

        binder_kinds = {LetDecl, VarDecl, AssignStmt}
        for kind in binder_kinds:
            assert kind in kinds, f"Expected {kind.__name__} to be visited"

    def test_walk_visits_all_expr_kinds(self) -> None:
        from agm.agl.syntax.visitor import walk

        prog = self._build_tree()
        visited: list[object] = []
        walk(prog, visited.append)
        kinds = {type(n) for n in visited}

        expr_kinds = {
            VarRef,
            FieldAccess,
            IndexAccess,
            NamedArg,
            Placeholder,
            BinaryOp,
            UnaryNot,
            UnaryNeg,
            IsTest,
            Call,
            Lambda,
            Block,
            If,
            IfBranch,
            Case,
            CaseBranch,
            Loop,
            Try,
            CatchClause,
            Raise,
            Break,
            Continue,
            UnitLit,
            IntLit,
            DecimalLit,
            BoolLit,
            NullLit,
            StringLit,
            AppliedT,
            ArrayLit,
            DictLit,
            DictEntry,
            Template,
            TextSegment,
            InterpSegment,
        }
        for kind in expr_kinds:
            assert kind in kinds, f"Expected {kind.__name__} to be visited"

    def test_walk_visits_typed_call_type_arg(self) -> None:
        from agm.agl.syntax.visitor import walk

        # A Call with ``type_args`` set: walk must descend into each type_arg so
        # the TypeExpr nodes are visited (covers the type_args traversal branch).
        callee = VarRef(name="ask-request", span=self._s(), node_id=700)
        type_arg = NameT(name="Review", span=self._s(), node_id=701)
        call = Call(
            callee=callee,
            args=(StringLit(value="q", span=self._s(), node_id=702),),
            named_args=(),
            type_args=(type_arg,),
            span=self._s(),
            node_id=703,
        )
        visited: list[object] = []
        walk(call, visited.append)
        assert type_arg in visited
        assert callee in visited

    def test_walk_visits_type_apply_children(self) -> None:
        from agm.agl.syntax.visitor import walk

        callee = VarRef(name="id", span=self._s(), node_id=710)
        type_arg = IntT(span=self._s(), node_id=711)
        node = TypeApply(expr=callee, type_args=(type_arg,), span=self._s(), node_id=712)
        visited: list[object] = []
        walk(node, visited.append)
        assert visited == [node, callee, type_arg]

    def test_walk_visits_let_pattern_and_every_child(self) -> None:
        from agm.agl.syntax.visitor import walk

        s = self._s()
        x = VarPattern(name="x", span=s, node_id=4)
        field = PatternField(name="left", pattern=x, span=s, node_id=3)
        pattern = AsPattern(
            pattern=ConstructorPattern(
                qualifier=None,
                name="Pair",
                positional=(),
                named=(field,),
                span=s,
                node_id=2,
            ),
            name="pair",
            span=s,
            node_id=5,
        )
        let = LetDecl(
            pattern=pattern,
            type_ann=IntT(span=s, node_id=6),
            value=IntLit(value=1, span=s, node_id=7),
            span=s,
            node_id=1,
        )
        visited: list[object] = []
        walk(let, visited.append)
        assert visited == [let, pattern, pattern.pattern, field, x, let.type_ann, let.value]

    def test_walk_visits_all_pattern_kinds(self) -> None:
        from agm.agl.syntax.visitor import walk

        prog = self._build_tree()
        visited: list[object] = []
        walk(prog, visited.append)
        kinds = {type(n) for n in visited}

        pattern_kinds = {
            WildcardPattern,
            LiteralPattern,
            VarPattern,
            AsPattern,
            ConstructorPattern,
            PatternField,
        }
        for kind in pattern_kinds:
            assert kind in kinds, f"Expected {kind.__name__} to be visited"

    def test_walk_visits_param(self) -> None:
        from agm.agl.syntax.visitor import walk

        prog = self._build_tree()
        visited: list[object] = []
        walk(prog, visited.append)
        kinds = {type(n) for n in visited}
        assert Param in kinds, "Expected Param to be visited"

    def test_walk_call_visits_callee_args_named_args(self) -> None:
        """walk(Call) visits callee, then each positional arg, then each named_arg."""
        from agm.agl.syntax.visitor import walk

        s = self._s()
        callee = VarRef(name="f", span=s, node_id=2)
        arg1 = IntLit(value=1, span=s, node_id=3)
        arg2 = IntLit(value=2, span=s, node_id=4)
        named_val = BoolLit(value=True, span=s, node_id=5)
        named = NamedArg(name="flag", value=named_val, span=s, node_id=6)
        call = Call(callee=callee, args=(arg1, arg2), named_args=(named,), span=s, node_id=1)

        visited: list[object] = []
        walk(call, visited.append)

        types_visited = [type(n) for n in visited]
        assert Call in types_visited
        assert VarRef in types_visited
        assert IntLit in types_visited
        assert NamedArg in types_visited
        assert BoolLit in types_visited

        # Order: Call, callee(VarRef), arg1, arg2, NamedArg, BoolLit
        assert visited[0] is call
        assert visited[1] is callee

    def test_walk_block_visits_all_items(self) -> None:
        """walk(Block) visits each item in order."""
        from agm.agl.syntax.visitor import walk

        s = self._s()
        a = IntLit(value=1, span=s, node_id=2)
        b = IntLit(value=2, span=s, node_id=3)
        blk = Block(items=(a, b), span=s, node_id=1)

        visited: list[object] = []
        walk(blk, visited.append)
        assert visited[0] is blk
        assert visited[1] is a
        assert visited[2] is b

    def test_walk_func_def_visits_params_return_body(self) -> None:
        """walk(FuncDef) visits each param, the return type, then the body."""
        from agm.agl.syntax.visitor import walk

        s = self._s()
        p_type = IntT(span=s, node_id=3)
        p = Param(name="x", type_expr=p_type, default=None, span=s, node_id=2)
        ret = TextT(span=s, node_id=4)
        body = VarRef(name="x", span=s, node_id=5)
        fd = FuncDef(name="f", params=(p,), return_type=ret, body=body, span=s, node_id=1)

        visited: list[object] = []
        walk(fd, visited.append)

        assert visited[0] is fd
        assert Param in {type(n) for n in visited}
        assert IntT in {type(n) for n in visited}
        assert TextT in {type(n) for n in visited}
        assert VarRef in {type(n) for n in visited}

    def test_walk_func_def_no_return_type(self) -> None:
        """walk(FuncDef) skips return_type when it is omitted."""
        from agm.agl.syntax.visitor import walk

        s = self._s()
        body = VarRef(name="x", span=s, node_id=5)
        fd = FuncDef(name="f", params=(), return_type=None, body=body, span=s, node_id=1)

        visited: list[object] = []
        walk(fd, visited.append)

        assert visited[0] is fd
        assert VarRef in {type(n) for n in visited}
        assert not any(isinstance(n, (IntT, TextT)) for n in visited)

    def test_walk_lambda_visits_params_return_type_body(self) -> None:
        """walk(Lambda) visits params, return_type (if present), then body."""
        from agm.agl.syntax.visitor import walk

        s = self._s()
        p_type = IntT(span=s, node_id=3)
        p = Param(name="x", type_expr=p_type, default=None, span=s, node_id=2)
        ret = BoolT(span=s, node_id=4)
        body = BoolLit(value=True, span=s, node_id=5)
        lam = Lambda(params=(p,), return_type=ret, body=body, span=s, node_id=1)

        visited: list[object] = []
        walk(lam, visited.append)

        kinds = {type(n) for n in visited}
        assert Lambda in kinds
        assert Param in kinds
        assert IntT in kinds
        assert BoolT in kinds
        assert BoolLit in kinds

    def test_walk_lambda_no_return_type(self) -> None:
        """walk(Lambda) skips return_type when None."""
        from agm.agl.syntax.visitor import walk

        s = self._s()
        body = UnitLit(span=s, node_id=2)
        lam = Lambda(params=(), return_type=None, body=body, span=s, node_id=1)

        visited: list[object] = []
        walk(lam, visited.append)
        kinds = {type(n) for n in visited}
        # Only Lambda and UnitLit should be visited (no type nodes)
        assert Lambda in kinds
        assert UnitLit in kinds

    def test_walk_if_visits_branches(self) -> None:
        """walk(If) visits each IfBranch; IfBranch visits cond and body."""
        from agm.agl.syntax.visitor import walk

        s = self._s()
        cond = BoolLit(value=True, span=s, node_id=3)
        body = IntLit(value=1, span=s, node_id=4)
        branch = IfBranch(cond=cond, body=body, span=s, node_id=2)
        node = If(branches=(branch,), span=s, node_id=1)

        visited: list[object] = []
        walk(node, visited.append)

        assert visited[0] is node
        assert visited[1] is branch
        assert visited[2] is cond
        assert visited[3] is body

    def test_walk_case_visits_subject_and_branches(self) -> None:
        """walk(Case) visits subject, then each CaseBranch (pattern + body)."""
        from agm.agl.syntax.visitor import walk

        s = self._s()
        subject = VarRef(name="x", span=s, node_id=2)
        pat = WildcardPattern(span=s, node_id=3)
        body = NullLit(span=s, node_id=4)
        branch = CaseBranch(pattern=pat, body=body, span=s, node_id=5)
        node = Case(subject=subject, branches=(branch,), span=s, node_id=1)

        visited: list[object] = []
        walk(node, visited.append)

        assert visited[0] is node
        assert visited[1] is subject
        assert visited[2] is branch
        assert visited[3] is pat
        assert visited[4] is body

    def test_walk_loop_visits_body_then_until_cond(self) -> None:
        """walk(Loop) visits body, then until_cond (when until_cond is set)."""
        from agm.agl.syntax.visitor import walk

        s = self._s()
        body = UnitLit(span=s, node_id=2)
        cond = BoolLit(value=True, span=s, node_id=3)
        node = Loop(
            for_var=None,
            for_iter=None,
            for_range_to=None,
            for_range_down=False,
            for_range_step=None,
            while_cond=None,
            bound=None,
            body=body,
            until_cond=cond,
            span=s,
            node_id=1,
        )

        visited: list[object] = []
        walk(node, visited.append)

        assert visited[0] is node
        assert visited[1] is body
        assert visited[2] is cond

    def test_walk_loop_done_visits_only_body(self) -> None:
        """walk(Loop with done/omitted) visits only body when until_cond is None."""
        from agm.agl.syntax.visitor import walk

        s = self._s()
        body = UnitLit(span=s, node_id=2)
        node = Loop(
            for_var=None,
            for_iter=None,
            for_range_to=None,
            for_range_down=False,
            for_range_step=None,
            while_cond=None,
            bound=None,
            body=body,
            until_cond=None,
            span=s,
            node_id=1,
        )

        visited: list[object] = []
        walk(node, visited.append)

        assert visited[0] is node
        assert visited[1] is body
        assert len(visited) == 2

    def test_walk_loop_with_bound_visits_bound_before_body(self) -> None:
        """walk(Loop) with bound visits the bound expression before body."""
        from agm.agl.syntax.visitor import walk

        s = self._s()
        bound = IntLit(value=5, span=s, node_id=4)
        body = UnitLit(span=s, node_id=2)
        cond = BoolLit(value=True, span=s, node_id=3)
        node = Loop(
            for_var=None,
            for_iter=None,
            for_range_to=None,
            for_range_down=False,
            for_range_step=None,
            while_cond=None,
            bound=bound,
            body=body,
            until_cond=cond,
            span=s,
            node_id=1,
        )

        visited: list[object] = []
        walk(node, visited.append)

        assert visited[0] is node
        assert visited[1] is bound
        assert visited[2] is body
        assert visited[3] is cond

    def test_walk_loop_with_for_iter_and_while_cond(self) -> None:
        """walk(Loop) with for_iter and while_cond visits them in order."""
        from agm.agl.syntax.visitor import walk

        s = self._s()
        for_iter = IntLit(value=0, span=s, node_id=5)
        while_cond = BoolLit(value=True, span=s, node_id=6)
        body = UnitLit(span=s, node_id=2)
        cond = BoolLit(value=False, span=s, node_id=3)
        node = Loop(
            for_var="i",
            for_iter=for_iter,
            for_range_to=None,
            for_range_down=False,
            for_range_step=None,
            while_cond=while_cond,
            bound=None,
            body=body,
            until_cond=cond,
            span=s,
            node_id=1,
        )

        visited: list[object] = []
        walk(node, visited.append)

        # Order: for_iter, while_cond, body, until_cond
        assert visited[0] is node
        assert visited[1] is for_iter
        assert visited[2] is while_cond
        assert visited[3] is body
        assert visited[4] is cond

    def test_walk_loop_range_for_visits_range_fields(self) -> None:
        """walk(Loop with range fields) visits start, range_to, range_step in order."""
        from agm.agl.syntax.visitor import walk

        s = self._s()
        start = IntLit(value=1, span=s, node_id=10)
        range_to = IntLit(value=10, span=s, node_id=11)
        range_step = IntLit(value=2, span=s, node_id=12)
        body = UnitLit(span=s, node_id=2)
        node = Loop(
            for_var="i",
            for_iter=start,
            for_range_to=range_to,
            for_range_down=False,
            for_range_step=range_step,
            while_cond=None,
            bound=None,
            body=body,
            until_cond=None,
            span=s,
            node_id=1,
        )

        visited: list[object] = []
        walk(node, visited.append)

        # Order: start (for_iter), range_to, range_step, body
        assert visited[0] is node
        assert visited[1] is start
        assert visited[2] is range_to
        assert visited[3] is range_step
        assert visited[4] is body

    def test_walk_try_visits_body_then_handlers(self) -> None:
        """walk(Try) visits body, then each CatchClause (which visits its body)."""
        from agm.agl.syntax.visitor import walk

        s = self._s()
        try_body = IntLit(value=1, span=s, node_id=2)
        handler_body = IntLit(value=0, span=s, node_id=3)
        clause = CatchClause(exc_type=None, binding=None, body=handler_body, span=s, node_id=4)
        node = Try(body=try_body, handlers=(clause,), span=s, node_id=1)

        visited: list[object] = []
        walk(node, visited.append)

        assert visited[0] is node
        assert visited[1] is try_body
        assert visited[2] is clause
        assert visited[3] is handler_body

    def test_walk_func_t_visits_param_types_then_result(self) -> None:
        """walk(FuncT) visits each param type then the result type."""
        from agm.agl.syntax.visitor import walk

        s = self._s()
        p1 = IntT(span=s, node_id=2)
        p2 = TextT(span=s, node_id=3)
        result = BoolT(span=s, node_id=4)
        ft = FuncT(params=(p1, p2), result=result, span=s, node_id=1)

        visited: list[object] = []
        walk(ft, visited.append)

        assert visited[0] is ft
        assert visited[1] is p1
        assert visited[2] is p2
        assert visited[3] is result

    def test_walk_param_with_default_visits_type_then_default(self) -> None:
        """walk(Param) visits type_expr then default (when not None)."""
        from agm.agl.syntax.visitor import walk

        s = self._s()
        p_type = IntT(span=s, node_id=2)
        default = IntLit(value=0, span=s, node_id=3)
        p = Param(name="x", type_expr=p_type, default=default, span=s, node_id=1)

        visited: list[object] = []
        walk(p, visited.append)

        assert visited[0] is p
        assert visited[1] is p_type
        assert visited[2] is default

    def test_walk_param_no_default_visits_type_only(self) -> None:
        """walk(Param) skips default when None."""
        from agm.agl.syntax.visitor import walk

        s = self._s()
        p_type = IntT(span=s, node_id=2)
        p = Param(name="x", type_expr=p_type, default=None, span=s, node_id=1)

        visited: list[object] = []
        walk(p, visited.append)

        assert visited == [p, p_type]

    def test_walk_return_visits_optional_value(self) -> None:
        from agm.agl.syntax.visitor import walk

        s = span()
        value = IntLit(value=1, span=s, node_id=2)
        node = Return(value=value, span=s, node_id=1)
        visited: list[object] = []
        walk(node, visited.append)
        assert visited == [node, value]

    def test_walk_bare_return_is_leaf(self) -> None:
        from agm.agl.syntax.visitor import walk

        node = Return(value=None, span=span(), node_id=1)
        visited: list[object] = []
        walk(node, visited.append)
        assert visited == [node]

    def test_walk_unit_lit_is_leaf(self) -> None:
        from agm.agl.syntax.visitor import walk

        node = UnitLit(span=span(), node_id=1)
        visited: list[object] = []
        walk(node, visited.append)
        assert visited == [node]

    def test_walk_unit_t_is_leaf(self) -> None:
        from agm.agl.syntax.visitor import walk

        node = UnitT(span=span(), node_id=1)
        visited: list[object] = []
        walk(node, visited.append)
        assert visited == [node]

    def test_walk_reaches_call_nodes(self) -> None:
        from agm.agl.syntax.visitor import walk

        s = self._s()
        callee = VarRef(name="f", span=s, node_id=2)
        call = Call(callee=callee, args=(), named_args=(), span=s, node_id=1)
        blk = Block(items=(call,), span=s, node_id=3)
        prog = Program(body=blk, span=s, node_id=0)

        visited: list[object] = []
        walk(prog, visited.append)
        assert [node for node in visited if isinstance(node, Call)] == [call]

    def test_walk_unknown_type_raises(self) -> None:
        """walk() on an unknown type should raise TypeError."""
        from agm.agl.syntax.visitor import walk

        class NotANode:
            pass

        with pytest.raises(TypeError):
            walk(NotANode(), lambda n: None)

    def test_walk_qual_var_ref_visits_qualifier_chain(self) -> None:
        """walk() on a slash-qualified VarRef visits its qualifier chain."""
        from agm.agl.parser import parse_program
        from agm.agl.syntax import QualifierChain
        from agm.agl.syntax.visitor import walk

        prog = parse_program("foo/bar::thing")
        visited: list[object] = []
        walk(prog, visited.append)
        assert any(isinstance(n, QualifierChain) for n in visited)

    def test_walk_qual_constructor_visits_qualifier_chain(self) -> None:
        """walk() on a slash-qualified reference visits its qualifier chain."""
        from agm.agl.parser import parse_program
        from agm.agl.syntax import QualifierChain
        from agm.agl.syntax.visitor import walk

        prog = parse_program("foo/bar::Color")
        visited: list[object] = []
        walk(prog, visited.append)
        assert any(isinstance(n, QualifierChain) for n in visited)

    @pytest.mark.parametrize(
        "source",
        (
            "let x: foo/bar::MyType = null",
            "let x: foo/bar::Box[int] = null",
            "case x of | m::Foo => 1",
        ),
    )
    def test_walk_qualified_references_visits_qualifier_chains(self, source: str) -> None:
        from agm.agl.parser import parse_program
        from agm.agl.syntax import QualifierChain
        from agm.agl.syntax.visitor import walk

        visited: list[object] = []
        walk(parse_program(source), visited.append)
        assert any(isinstance(node, QualifierChain) for node in visited)

    def test_walk_constructor_pattern_visits_positional(self) -> None:
        """walk() on a ConstructorPattern with positional sub-patterns visits each one."""
        from agm.agl.parser import parse_program
        from agm.agl.syntax.visitor import walk

        prog = parse_program("case x of | Some(v) => 1")
        visited: list[object] = []
        walk(prog, visited.append)
        var_pats = [n for n in visited if isinstance(n, VarPattern)]
        assert any(vp.name == "v" for vp in var_pats), (
            "Positional sub-pattern VarPattern not visited in ConstructorPattern"
        )

    def test_walk_known_node_without_branch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A node in _KNOWN_NODE_TYPES but lacking a walk branch must fail loudly.

        Guards the lockstep invariant: a future contributor who adds a node to
        _KNOWN_NODE_TYPES without an isinstance branch in walk() should get a
        crash, not silently-dropped children.
        """
        from agm.agl.syntax import visitor

        class FakeKnownNode:
            pass

        monkeypatch.setattr(
            visitor,
            "_KNOWN_NODE_TYPES",
            visitor._KNOWN_NODE_TYPES | {FakeKnownNode},
        )

        with pytest.raises(AssertionError, match="known but has no walk branch"):
            visitor.walk(FakeKnownNode(), lambda n: None)


# ---------------------------------------------------------------------------
# Union alias sanity
# ---------------------------------------------------------------------------


class TestUnionAliases:
    """Verify that the union aliases contain the right members."""

    def test_var_ref_is_expr(self) -> None:
        import typing

        args = typing.get_args(Expr)
        assert VarRef in args

    def test_raise_is_expr(self) -> None:
        import typing

        args = typing.get_args(Expr)
        assert Raise in args, "Raise must be in Expr (bottom type)"
        assert Return in args, "Return must be in Expr (bottom type)"

    def test_block_is_expr(self) -> None:
        import typing

        args = typing.get_args(Expr)
        assert Block in args

    def test_if_is_expr(self) -> None:
        import typing

        args = typing.get_args(Expr)
        assert If in args

    def test_case_is_expr(self) -> None:
        import typing

        args = typing.get_args(Expr)
        assert Case in args

    def test_loop_is_expr(self) -> None:
        import typing

        args = typing.get_args(Expr)
        assert Loop in args

    def test_try_is_expr(self) -> None:
        import typing

        args = typing.get_args(Expr)
        assert Try in args

    def test_call_is_expr(self) -> None:
        import typing

        args = typing.get_args(Expr)
        assert Call in args

    def test_placeholder_is_expr(self) -> None:
        import typing

        args = typing.get_args(Expr)
        assert Placeholder in args

    def test_index_access_is_expr(self) -> None:
        import typing

        args = typing.get_args(Expr)
        assert IndexAccess in args

    def test_lambda_is_expr(self) -> None:
        import typing

        args = typing.get_args(Expr)
        assert Lambda in args

    def test_unit_lit_is_expr(self) -> None:
        import typing

        args = typing.get_args(Expr)
        assert UnitLit in args

    def test_as_pattern_is_pattern(self) -> None:
        import typing

        args = typing.get_args(Pattern)
        assert AsPattern in args

    def test_wildcard_pattern_is_pattern(self) -> None:
        import typing

        args = typing.get_args(Pattern)
        assert WildcardPattern in args

    def test_text_segment_is_template_segment(self) -> None:
        import typing

        args = typing.get_args(TemplateSegment)
        assert TextSegment in args

    def test_closed_pattern_binder_and_item_unions(self) -> None:
        import typing

        assert set(typing.get_args(Pattern)) == {
            WildcardPattern,
            LiteralPattern,
            VarPattern,
            AsPattern,
            ConstructorPattern,
        }
        assert set(typing.get_args(Binder)) == {LetDecl, VarDecl, AssignStmt}
        assert set(typing.get_args(Item)) == {
            *typing.get_args(Declaration),
            *typing.get_args(Binder),
            *typing.get_args(Expr),
            ScopeRegion,
        }

    def test_assign_target_union_members(self) -> None:
        import typing

        args = typing.get_args(AssignTarget)
        assert NameTarget in args
        assert IndexTarget in args
        assert FieldTarget in args

    def test_item_contains_declaration_binder_expr(self) -> None:
        import typing

        args = typing.get_args(Item)
        # Item is Declaration | Binder | Expr — check a sample from each
        # (FuncDef is a Declaration, LetDecl is a Binder, VarRef is an Expr)
        assert FuncDef in args
        assert LetDecl in args
        assert VarRef in args


# ---------------------------------------------------------------------------
# Removed nodes are truly gone
# ---------------------------------------------------------------------------


class TestRemovedNodes:
    """Verify that legacy nodes are no longer part of the public API."""

    @pytest.mark.parametrize(
        "name",
        [
            "AgentCall",
            "PassStmt",
            "PrintStmt",
            "ExprStmt",
            "DoUntil",
            "IfStmt",
            "CaseStmt",
            "CaseExpr",
            "IfExpr",
            "TryCatch",
            "CallOptions",
            "Constructor",
            "Stmt",
        ],
    )
    def test_v1_node_not_exported(self, name: str) -> None:
        module = importlib.import_module("agm.agl.syntax")
        assert not hasattr(module, name)


# ---------------------------------------------------------------------------
# Cast node
# ---------------------------------------------------------------------------


class TestCastNode:
    """Tests for the Cast AST node."""

    def _sp(self) -> SourceSpan:
        return span()

    def test_cast_in_expr_union(self) -> None:
        import typing

        assert Cast in typing.get_args(Expr)

    def test_cast_expr_construction(self) -> None:
        expr = IntLit(value=1, span=self._sp(), node_id=0)
        target = IntT(span=self._sp(), node_id=1)
        node = Cast(expr=expr, target_type=target, test_only=False, span=self._sp(), node_id=2)
        assert node.expr is expr
        assert node.target_type is target
        assert node.test_only is False

    def test_cast_test_construction(self) -> None:
        expr = VarRef(name="x", span=self._sp(), node_id=0)
        target = TextT(span=self._sp(), node_id=1)
        node = Cast(expr=expr, target_type=target, test_only=True, span=self._sp(), node_id=2)
        assert node.test_only is True

    def test_cast_equality_ignores_span_and_node_id(self) -> None:
        expr = IntLit(value=42, span=self._sp(), node_id=0)
        target = TextT(span=self._sp(), node_id=1)
        n1 = Cast(
            expr=expr,
            target_type=target,
            test_only=False,
            span=SourceSpan(1, 0, 1, 5, 0, 5),
            node_id=99,
        )
        n2 = Cast(
            expr=expr,
            target_type=target,
            test_only=False,
            span=SourceSpan(2, 0, 2, 5, 0, 5),
            node_id=100,
        )
        assert n1 == n2

    def test_cast_immutable(self) -> None:
        expr = NullLit(span=self._sp(), node_id=0)
        target = BoolT(span=self._sp(), node_id=1)
        node = Cast(expr=expr, target_type=target, test_only=False, span=self._sp(), node_id=2)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(node, "test_only", True)

    def test_cast_target_uses_type_expr(self) -> None:
        """target_type accepts any TypeExpr, including generic forms."""
        expr = VarRef(name="xs", span=self._sp(), node_id=0)
        target = ArrayT(elem=IntT(span=self._sp(), node_id=1), span=self._sp(), node_id=2)
        node = Cast(expr=expr, target_type=target, test_only=False, span=self._sp(), node_id=3)
        assert isinstance(node.target_type, ArrayT)

    def test_cast_walk_visits_cast_and_children(self) -> None:
        """walk() visits Cast, its expr child, and its target_type child."""
        from agm.agl.syntax.visitor import walk

        expr = IntLit(value=1, span=self._sp(), node_id=0)
        target = TextT(span=self._sp(), node_id=1)
        node = Cast(expr=expr, target_type=target, test_only=False, span=self._sp(), node_id=2)
        visited: list[object] = []
        walk(node, visited.append)
        assert any(isinstance(n, Cast) for n in visited)
        assert any(isinstance(n, IntLit) for n in visited)
        assert any(isinstance(n, TextT) for n in visited)


# ---------------------------------------------------------------------------
# Module system nodes
# ---------------------------------------------------------------------------


class TestModuleSystemNodes:
    """Tests for import, use, and export AST nodes."""

    def _sp(self) -> SourceSpan:
        return SourceSpan(1, 0, 1, 1, 0, 1)

    def test_import_decl_preserves_tail_and_hidden_items(self) -> None:
        from agm.agl.syntax import ImportDecl, ImportItem

        item = ImportItem(name="member", rename="renamed", span=self._sp(), node_id=1)
        hidden = ImportItem(name="secret", rename=None, span=self._sp(), node_id=2)
        decl = ImportDecl(
            module_path=("foo", "bar"),
            wildcard=False,
            alias=None,
            tail=(item,),
            hidden=(hidden,),
            span=self._sp(),
            node_id=0,
        )
        assert decl.tail == (item,)
        assert decl.hidden == (hidden,)

    def test_use_decl_preserves_target_and_selection(self) -> None:
        from agm.agl.syntax import ImportItem, ScopeSegment, UseDecl

        target = ScopeSegment(name="Scope", span=self._sp(), node_id=1)
        item = ImportItem(name="member", rename=None, span=self._sp(), node_id=2)
        decl = UseDecl(
            anchored=False,
            target=(target,),
            tail=(item,),
            hidden=(),
            alias=None,
            span=self._sp(),
            node_id=0,
        )
        assert decl.target == (target,)
        assert decl.tail == (item,)

    def test_parsed_use_preserves_current_module_anchor(self) -> None:
        from agm.agl.parser import parse_program
        from agm.agl.syntax import UseDecl

        program = parse_program("use ::Scope::*\nuse ::Scope as S")
        glob, alias = program.body.items
        assert isinstance(glob, UseDecl)
        assert isinstance(alias, UseDecl)
        assert glob.anchored is True
        assert alias.anchored is True
        assert tuple(segment.name for segment in glob.target) == ("Scope",)
        assert glob.tail == ()
        assert alias.alias == "S"
        assert alias.tail is None

    def test_module_declarations_walk_selection_items(self) -> None:
        from agm.agl.syntax import ExportDecl, ExportItem, ImportDecl, ImportItem
        from agm.agl.syntax.visitor import walk

        import_item = ImportItem(name="x", rename=None, span=self._sp(), node_id=1)
        export_item = ExportItem(name="x", rename=None, span=self._sp(), node_id=3)
        import_decl = ImportDecl(
            module_path=("m",),
            wildcard=False,
            alias=None,
            tail=(import_item,),
            hidden=(),
            span=self._sp(),
            node_id=0,
        )
        export_decl = ExportDecl(
            module_path=("m",),
            wildcard=False,
            items=(export_item,),
            hidden=(),
            span=self._sp(),
            node_id=2,
        )
        visited: list[object] = []
        walk(import_decl, visited.append)
        walk(export_decl, visited.append)
        assert import_item in visited
        assert export_item in visited

    def test_infix_decl_walk_is_leaf(self) -> None:
        from agm.agl.syntax import InfixAssoc, InfixDecl
        from agm.agl.syntax.visitor import walk

        decl = InfixDecl(
            name="|>",
            assoc=InfixAssoc.LEFT,
            priority=12,
            priority_base=None,
            priority_delta=0,
            span=self._sp(),
            node_id=0,
        )
        visited: list[object] = []
        walk(decl, visited.append)
        assert visited == [decl]

    def test_raw_infix_chain_walk_visits_all_raw_nodes(self) -> None:
        from agm.agl.syntax import RawInfixChain, RawInfixOperand, RawInfixOperator, RawPrefixNot
        from agm.agl.syntax.visitor import walk

        raw_not = RawPrefixNot(span=self._sp(), node_id=1)
        operand = RawInfixOperand(
            expr=IntLit(value=1, span=self._sp(), node_id=2),
            prefix_nots=(raw_not,),
            span=self._sp(),
            node_id=3,
        )
        operator = RawInfixOperator(
            name="|>",
            callee_node_id=4,
            span=self._sp(),
            node_id=5,
        )
        chain = RawInfixChain(
            operands=(operand,),
            operators=(operator,),
            span=self._sp(),
            node_id=6,
        )
        visited: list[object] = []
        walk(chain, visited.append)

        assert visited == [chain, operand, operand.expr, raw_not, operator]

    def test_var_ref_qualifier_chain_default_none(self) -> None:
        ref = VarRef(name="x", span=self._sp(), node_id=0)
        assert ref.qualifier is None

    def test_var_ref_with_qualifier_chain(self) -> None:
        from agm.agl.syntax import QualifierChain, QualifierSegment

        segment = QualifierSegment(name="foo", type_args=None, span=self._sp(), node_id=0)
        chain = QualifierChain(
            anchor=None, segments=(segment,), member="x", span=self._sp(), node_id=1
        )
        ref = VarRef(name="x", span=self._sp(), node_id=2, qualifier=chain)
        assert ref.qualifier is chain

    def test_qualified_reference_defaults_to_none(self) -> None:
        t = NameT(name="MyType", span=self._sp(), node_id=0)
        pat = ConstructorPattern(name="Foo", positional=(), named=(), span=self._sp(), node_id=0)
        assert t.qualifier is None
        assert pat.qualifier is None


# ---------------------------------------------------------------------------
# Declaration attributes
# ---------------------------------------------------------------------------


class TestDeclarationAttributes:
    """Every defining declaration carries a raw attribute prefix."""

    def _sp(self) -> SourceSpan:
        return span()

    def _attribute(self, name: str = "doc") -> Attribute:
        return Attribute(
            name=name,
            args=(StringLit(value="why", span=self._sp(), node_id=nid()),),
            named_args=(),
            span=self._sp(),
            node_id=nid(),
        )

    def _declarations(self, attributes: tuple[Attribute, ...] | None = None) -> dict[str, object]:
        """Build one of every attributed node.

        With no ``attributes`` argument the nodes are constructed without the
        keyword at all, so the dataclass defaults are what the caller sees.
        """
        sp = self._sp()
        body = IntLit(value=1, span=sp, node_id=nid())
        type_expr = IntT(span=sp, node_id=nid())
        prefix = {} if attributes is None else {"attributes": attributes}
        field = Param(
            name="x",
            type_expr=type_expr,
            default=None,
            span=sp,
            node_id=nid(),
        )
        return {
            "Param": Param(
                name="x",
                type_expr=type_expr,
                default=None,
                span=sp,
                node_id=nid(),
                **prefix,
            ),
            "FuncDef": FuncDef(
                name="f",
                params=(),
                return_type=None,
                body=body,
                span=sp,
                node_id=nid(),
                **prefix,
            ),
            "RecordDef": RecordDef(name="R", fields=(field,), span=sp, node_id=nid(), **prefix),
            "VariantDef": VariantDef(name="M", fields=(field,), span=sp, node_id=nid(), **prefix),
            "EnumDef": EnumDef(name="E", members=(), span=sp, node_id=nid(), **prefix),
            "ExceptionDef": ExceptionDef(
                name="X",
                fields=(field,),
                base=None,
                span=sp,
                node_id=nid(),
                **prefix,
            ),
            "TypeAlias": TypeAlias(name="T", type_expr=type_expr, span=sp, node_id=nid(), **prefix),
            "LetDecl": LetDecl(
                pattern=VarPattern(name="x", span=sp, node_id=nid()),
                type_ann=None,
                value=body,
                span=sp,
                node_id=nid(),
                **prefix,
            ),
            "VarDecl": VarDecl(
                name="x",
                type_ann=None,
                value=body,
                span=sp,
                node_id=nid(),
                **prefix,
            ),
            "BuiltinVarDecl": BuiltinVarDecl(
                name="x", type_ann=type_expr, span=sp, node_id=nid(), **prefix
            ),
        }

    def test_every_defining_declaration_defaults_to_no_attributes(self) -> None:
        for name, node in self._declarations().items():
            assert node.attributes == (), name

    def test_every_defining_declaration_keeps_the_attributes_it_was_given(self) -> None:
        attributes = (self._attribute(),)
        for name, node in self._declarations(attributes).items():
            assert node.attributes == attributes, name

    def test_attribute_holds_its_name_and_raw_arguments(self) -> None:
        value = StringLit(value="python-name", span=self._sp(), node_id=nid())
        named = NamedArg(name="short", value=value, span=self._sp(), node_id=nid())
        attribute = Attribute(
            name="extern-name",
            args=(value,),
            named_args=(named,),
            span=self._sp(),
            node_id=nid(),
        )
        assert attribute.name == "extern-name"
        assert attribute.args == (value,)
        assert attribute.named_args == (named,)

    def test_attribute_equality_ignores_span_and_node_id(self) -> None:
        first = Attribute(name="arg-pos", args=(), named_args=(), span=span(1, 0, 1, 8), node_id=1)
        second = Attribute(
            name="arg-pos", args=(), named_args=(), span=span(9, 0, 9, 8), node_id=99
        )
        assert first == second

    def test_attribute_is_immutable(self) -> None:
        attribute = self._attribute()
        with pytest.raises(FrozenInstanceError):
            setattr(attribute, "name", "other")

    def test_walk_visits_attribute_arguments_of_every_declaration(self) -> None:
        from agm.agl.syntax.visitor import walk

        for name, node in self._declarations((self._attribute(),)).items():
            visited: list[object] = []
            walk(node, visited.append)
            assert any(isinstance(seen, Attribute) for seen in visited), name
            assert any(isinstance(seen, StringLit) and seen.value == "why" for seen in visited), (
                name
            )

    def test_walk_visits_attribute_named_arguments(self) -> None:
        from agm.agl.syntax.visitor import walk

        value = StringLit(value="named", span=self._sp(), node_id=nid())
        attribute = Attribute(
            name="doc",
            args=(),
            named_args=(NamedArg(name="text", value=value, span=self._sp(), node_id=nid()),),
            span=self._sp(),
            node_id=nid(),
        )
        visited: list[object] = []
        walk(attribute, visited.append)
        assert visited[0] is attribute
        assert value in visited

    def test_walk_reaches_declaration_attributes(self) -> None:
        from agm.agl.syntax.visitor import walk

        visited: list[object] = []
        walk(
            FuncDef(
                name="f",
                params=(),
                return_type=None,
                body=IntLit(value=1, span=self._sp(), node_id=nid()),
                span=self._sp(),
                node_id=nid(),
                attributes=(self._attribute("arg-named"),),
            ),
            visited.append,
        )
        assert [node.name for node in visited if isinstance(node, Attribute)] == ["arg-named"]
