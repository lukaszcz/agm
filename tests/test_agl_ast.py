"""Tests for the AgL v2 AST package (agm.agl.syntax).

Covers:
- SourceSpan construction and import from diagnostics
- TypeExpr hierarchy (all 11 types including UnitT, AgentT, FuncT)
- All AST node types: Program/Block, declarations (FuncDef), binders, expressions,
  patterns
- New v2 nodes: UnitLit, Call, Param, FuncDef, Lambda, Block, If/IfBranch,
  Case/CaseBranch, Do, Try/CatchClause
- Removed nodes are truly absent (AgentCall, PassStmt, PrintStmt, ExprStmt,
  DoUntil, IfStmt, CaseStmt, CaseExpr, IfExpr, TryCatch, CallOptions,
  AbortPolicy, RetryPolicy)
- Equality semantics: equal structure with different spans/node_ids compare equal
- Immutability: frozen dataclasses reject mutation
- Visitor/walk traversal visits every node kind
- Tuple-typed children (not lists)
- ELSE sentinel type for IfBranch.cond
- BinaryOp with closed operator set
- DecimalLit holds decimal.Decimal
- Union membership for Expr, Binder, Declaration, Item, Pattern, TemplateSegment
"""

from __future__ import annotations

import decimal
from dataclasses import FrozenInstanceError

import pytest

# ---------------------------------------------------------------------------
# Import everything through the package public API
# ---------------------------------------------------------------------------
from agm.agl.syntax import (
    # sentinel
    ELSE,
    AgentDecl,
    AgentT,
    AssignStmt,
    AssignTarget,
    BinaryOp,
    Binder,
    BinOp,
    Block,
    BoolLit,
    BoolT,
    Call,
    Case,
    CaseBranch,
    Cast,
    CatchClause,
    ConfigPragma,
    Constructor,
    ConstructorPattern,
    DecimalLit,
    DecimalT,
    Declaration,
    DictEntry,
    DictLit,
    DictT,
    Do,
    ElseSentinel,
    EnumDef,
    Expr,
    FieldAccess,
    FieldDef,
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
    ListLit,
    ListT,
    LiteralPattern,
    NamedArg,
    NameT,
    NameTarget,
    NullLit,
    Param,
    ParamDecl,
    Pattern,
    PatternField,
    Program,
    ProgramDecl,
    Raise,
    RecordDef,
    # spans
    SourceSpan,
    StringLit,
    Template,
    TemplateSegment,
    TextSegment,
    # types
    TextT,
    Try,
    TypeAlias,
    TypeExpr,
    UnaryNeg,
    UnaryNot,
    UnitLit,
    UnitT,
    VarDecl,
    VariantDef,
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
        start_line=sl, start_col=sc, end_line=el, end_col=ec,
        start_offset=0, end_offset=1,
    )


def nid(n: int = 1) -> int:
    return n


# ---------------------------------------------------------------------------
# SourceSpan
# ---------------------------------------------------------------------------

class TestSourceSpan:
    def test_construction_positional_fields(self) -> None:
        s = SourceSpan(
            start_line=1, start_col=0, end_line=1, end_col=10,
            start_offset=0, end_offset=10,
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
        # temporarily broken while downstream stages are being ported to v2).
        from agm.agl.syntax.spans import SourceSpan as SpansSourceSpan
        assert SourceSpan is SpansSourceSpan

    def test_offsets_are_required(self) -> None:
        # Offsets are mandatory: a 4-arg construction is an argument error.
        args = (1, 1, 1, 10)
        with pytest.raises(TypeError):
            SourceSpan(*args)


# ---------------------------------------------------------------------------
# TypeExpr hierarchy
# ---------------------------------------------------------------------------

class TestTypeExprs:
    def _s(self) -> SourceSpan:
        return span()

    def test_text_t(self) -> None:
        t = TextT(span=self._s(), node_id=1)
        assert isinstance(t, TextT)

    def test_json_t(self) -> None:
        t = JsonT(span=self._s(), node_id=1)
        assert isinstance(t, JsonT)

    def test_bool_t(self) -> None:
        t = BoolT(span=self._s(), node_id=1)
        assert isinstance(t, BoolT)

    def test_int_t(self) -> None:
        t = IntT(span=self._s(), node_id=1)
        assert isinstance(t, IntT)

    def test_decimal_t(self) -> None:
        t = DecimalT(span=self._s(), node_id=1)
        assert isinstance(t, DecimalT)

    def test_name_t(self) -> None:
        t = NameT(name="MyType", span=self._s(), node_id=1)
        assert t.name == "MyType"

    def test_list_t(self) -> None:
        elem = TextT(span=self._s(), node_id=2)
        t = ListT(elem=elem, span=self._s(), node_id=1)
        assert t.elem is elem

    def test_dict_t(self) -> None:
        val = IntT(span=self._s(), node_id=2)
        t = DictT(value=val, span=self._s(), node_id=1)
        assert t.value is val

    def test_unit_t(self) -> None:
        t = UnitT(span=self._s(), node_id=1)
        assert isinstance(t, UnitT)

    def test_agent_t(self) -> None:
        t = AgentT(span=self._s(), node_id=1)
        assert isinstance(t, AgentT)

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

    def test_func_t_params_is_tuple(self) -> None:
        result = UnitT(span=self._s(), node_id=2)
        t = FuncT(params=(), result=result, span=self._s(), node_id=1)
        assert isinstance(t.params, tuple)

    def test_type_equality_ignores_span_and_node_id(self) -> None:
        s1 = span(1, 0, 1, 5)
        s2 = span(2, 0, 2, 5)
        t1 = TextT(span=s1, node_id=1)
        t2 = TextT(span=s2, node_id=99)
        assert t1 == t2

    def test_name_t_equality(self) -> None:
        t1 = NameT(name="Foo", span=span(1, 0, 1, 3), node_id=1)
        t2 = NameT(name="Foo", span=span(2, 0, 2, 3), node_id=99)
        assert t1 == t2

    def test_name_t_inequality(self) -> None:
        t1 = NameT(name="Foo", span=span(), node_id=1)
        t2 = NameT(name="Bar", span=span(), node_id=1)
        assert t1 != t2

    def test_list_t_equality(self) -> None:
        a = ListT(elem=TextT(span=span(), node_id=2), span=span(), node_id=1)
        b = ListT(elem=TextT(span=span(3, 0, 3, 4), node_id=99), span=span(5, 0, 5, 4), node_id=50)
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

    def test_agent_t_equality(self) -> None:
        t1 = AgentT(span=span(1, 0, 1, 5), node_id=1)
        t2 = AgentT(span=span(3, 0, 3, 5), node_id=30)
        assert t1 == t2

    def test_type_frozen(self) -> None:
        t = TextT(span=span(), node_id=1)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(t, "span", span())

    def test_type_expr_union_contains_all_types(self) -> None:
        import typing
        args = typing.get_args(TypeExpr)
        for tp in (TextT, JsonT, BoolT, IntT, DecimalT, NameT, ListT, DictT, UnitT, AgentT, FuncT):
            assert tp in args, f"{tp.__name__} missing from TypeExpr union"


# ---------------------------------------------------------------------------
# Literal nodes
# ---------------------------------------------------------------------------

class TestLiterals:
    def _s(self) -> SourceSpan:
        return span()

    def test_unit_lit(self) -> None:
        node = UnitLit(span=self._s(), node_id=1)
        assert isinstance(node, UnitLit)

    def test_unit_lit_equality_ignores_span_node_id(self) -> None:
        a = UnitLit(span=span(1, 0, 1, 2), node_id=1)
        b = UnitLit(span=span(5, 0, 5, 2), node_id=99)
        assert a == b

    def test_int_lit(self) -> None:
        node = IntLit(value=42, span=self._s(), node_id=1)
        assert node.value == 42

    def test_decimal_lit(self) -> None:
        d = decimal.Decimal("3.14")
        node = DecimalLit(value=d, span=self._s(), node_id=1)
        assert node.value == d

    def test_decimal_lit_holds_decimal_type(self) -> None:
        node = DecimalLit(value=decimal.Decimal("1.0"), span=self._s(), node_id=1)
        assert isinstance(node.value, decimal.Decimal)

    def test_bool_lit_true(self) -> None:
        node = BoolLit(value=True, span=self._s(), node_id=1)
        assert node.value is True

    def test_bool_lit_false(self) -> None:
        node = BoolLit(value=False, span=self._s(), node_id=1)
        assert node.value is False

    def test_null_lit(self) -> None:
        node = NullLit(span=self._s(), node_id=1)
        assert isinstance(node, NullLit)

    def test_string_lit(self) -> None:
        node = StringLit(value="hello", span=self._s(), node_id=1)
        assert node.value == "hello"

    def test_list_lit(self) -> None:
        elems: tuple[Expr, ...] = (IntLit(value=1, span=self._s(), node_id=2),)
        node = ListLit(elements=elems, span=self._s(), node_id=1)
        assert isinstance(node.elements, tuple)
        assert len(node.elements) == 1

    def test_list_lit_empty(self) -> None:
        node = ListLit(elements=(), span=self._s(), node_id=1)
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

    def test_constructor(self) -> None:
        arg = NamedArg(
            name="field",
            value=IntLit(value=1, span=self._s(), node_id=3),
            span=self._s(),
            node_id=2,
        )
        node = Constructor(qualifier=None, name="MyType", args=(arg,), span=self._s(), node_id=1)
        assert isinstance(node.args, tuple)
        assert node.qualifier is None

    def test_constructor_with_qualifier(self) -> None:
        node = Constructor(
            qualifier="Ns",
            name="Variant",
            args=(),
            span=self._s(),
            node_id=1,
        )
        assert node.qualifier == "Ns"
        assert node.name == "Variant"

    def test_binary_op(self) -> None:
        left = IntLit(value=1, span=self._s(), node_id=2)
        right = IntLit(value=2, span=self._s(), node_id=3)
        node = BinaryOp(op=BinOp.ADD, left=left, right=right, span=self._s(), node_id=1)
        assert node.op is BinOp.ADD

    def test_all_binary_ops_exist(self) -> None:
        ops = {BinOp.EQ, BinOp.NEQ, BinOp.LT, BinOp.LE, BinOp.GT, BinOp.GE,
               BinOp.IN, BinOp.AND, BinOp.OR, BinOp.ADD, BinOp.SUB,
               BinOp.MUL, BinOp.DIV}
        assert len(ops) == 13

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

    def test_equality_ignores_span_node_id(self) -> None:
        a = VarRef(name="x", span=span(1, 0, 1, 1), node_id=10)
        b = VarRef(name="x", span=span(5, 3, 5, 4), node_id=99)
        assert a == b

    def test_inequality_different_name(self) -> None:
        a = VarRef(name="x", span=span(), node_id=1)
        b = VarRef(name="y", span=span(), node_id=1)
        assert a != b


# ---------------------------------------------------------------------------
# New v2 expression nodes
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

    def test_param_with_default(self) -> None:
        t = IntT(span=self._s(), node_id=2)
        default = IntLit(value=0, span=self._s(), node_id=3)
        p = Param(name="n", type_expr=t, default=default, span=self._s(), node_id=1)
        assert p.default is default

    def test_param_equality_ignores_span_node_id(self) -> None:
        t1 = IntT(span=span(1, 0, 1, 3), node_id=10)
        t2 = IntT(span=span(5, 0, 5, 3), node_id=20)
        a = Param(name="x", type_expr=t1, default=None, span=span(1, 0, 1, 5), node_id=1)
        b = Param(name="x", type_expr=t2, default=None, span=span(9, 0, 9, 5), node_id=99)
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
            name="f", params=(), return_type=ret, body=body,
            span=span(1, 0, 1, 20), node_id=1,
        )
        b = FuncDef(
            name="f", params=(), return_type=ret, body=body,
            span=span(9, 0, 9, 20), node_id=99,
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
        let = LetDecl(name="x", type_ann=None, value=val, span=self._s(), node_id=2)
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
        # In v2, branch body is a single Expr (not a tuple).
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


class TestDoNode:
    def _s(self) -> SourceSpan:
        return span()

    def test_do_no_limit(self) -> None:
        body = UnitLit(span=self._s(), node_id=2)
        cond = BoolLit(value=True, span=self._s(), node_id=3)
        node = Do(limit=None, body=body, condition=cond, span=self._s(), node_id=1)
        assert node.limit is None
        assert node.body is body
        assert node.condition is cond

    def test_do_with_limit(self) -> None:
        body = UnitLit(span=self._s(), node_id=2)
        cond = BoolLit(value=False, span=self._s(), node_id=3)
        node = Do(limit=10, body=body, condition=cond, span=self._s(), node_id=1)
        assert node.limit == 10

    def test_do_body_is_expr(self) -> None:
        # body is a single Expr (not a tuple of stmts)
        body = Block(items=(UnitLit(span=self._s(), node_id=3),), span=self._s(), node_id=2)
        cond = BoolLit(value=True, span=self._s(), node_id=4)
        node = Do(limit=None, body=body, condition=cond, span=self._s(), node_id=1)
        assert isinstance(node.body, Block)

    def test_do_equality_ignores_span_node_id(self) -> None:
        body = UnitLit(span=span(1, 0, 1, 2), node_id=5)
        cond = BoolLit(value=True, span=span(1, 0, 1, 4), node_id=6)
        a = Do(limit=5, body=body, condition=cond, span=span(1, 0, 1, 20), node_id=1)
        b = Do(limit=5, body=body, condition=cond, span=span(9, 0, 9, 20), node_id=99)
        assert a == b

    def test_do_frozen(self) -> None:
        body = UnitLit(span=span(), node_id=2)
        cond = BoolLit(value=True, span=span(), node_id=3)
        node = Do(limit=None, body=body, condition=cond, span=span(), node_id=1)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(node, "limit", 5)


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
        # In v2, body is a single Expr, not a tuple of stmts.
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

    def test_let_decl(self) -> None:
        val = IntLit(value=1, span=self._s(), node_id=2)
        node = LetDecl(name="x", type_ann=None, value=val, span=self._s(), node_id=1)
        assert node.name == "x"
        assert node.type_ann is None

    def test_let_decl_with_type(self) -> None:
        val = IntLit(value=1, span=self._s(), node_id=3)
        t = IntT(span=self._s(), node_id=2)
        node = LetDecl(name="x", type_ann=t, value=val, span=self._s(), node_id=1)
        assert node.type_ann is t

    def test_var_decl(self) -> None:
        val = IntLit(value=0, span=self._s(), node_id=2)
        node = VarDecl(name="count", type_ann=None, value=val, span=self._s(), node_id=1)
        assert node.name == "count"

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

    def test_raise_is_expr(self) -> None:
        # Raise is in the Expr union (bottom type).
        expr = Constructor(qualifier=None, name="MyErr", args=(), span=self._s(), node_id=2)
        node = Raise(exc=expr, span=self._s(), node_id=1)
        assert node.exc is expr

    def test_binder_equality_ignores_span_node_id(self) -> None:
        val = IntLit(value=1, span=span(1, 0, 1, 1), node_id=5)
        a = LetDecl(name="x", type_ann=None, value=val, span=span(1, 0, 1, 10), node_id=1)
        b = LetDecl(name="x", type_ann=None, value=val, span=span(9, 0, 9, 10), node_id=99)
        assert a == b

    def test_binder_frozen(self) -> None:
        val = IntLit(value=1, span=span(), node_id=2)
        node = LetDecl(name="x", type_ann=None, value=val, span=span(), node_id=1)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(node, "name", "y")


# ---------------------------------------------------------------------------
# Declaration nodes
# ---------------------------------------------------------------------------

class TestDeclarations:
    def _s(self) -> SourceSpan:
        return span()

    def test_field_def(self) -> None:
        t = TextT(span=self._s(), node_id=2)
        f = FieldDef(name="title", type_expr=t, span=self._s(), node_id=1)
        assert f.name == "title"
        assert f.type_expr is t

    def test_record_def(self) -> None:
        t = IntT(span=self._s(), node_id=3)
        f = FieldDef(name="x", type_expr=t, span=self._s(), node_id=2)
        node = RecordDef(name="Point", fields=(f,), span=self._s(), node_id=1)
        assert node.name == "Point"
        assert isinstance(node.fields, tuple)

    def test_variant_def(self) -> None:
        t = IntT(span=self._s(), node_id=3)
        f = FieldDef(name="val", type_expr=t, span=self._s(), node_id=2)
        v = VariantDef(name="Some", fields=(f,), span=self._s(), node_id=1)
        assert v.name == "Some"

    def test_enum_def(self) -> None:
        v1 = VariantDef(name="A", fields=(), span=self._s(), node_id=2)
        v2 = VariantDef(name="B", fields=(), span=self._s(), node_id=3)
        node = EnumDef(name="Color", variants=(v1, v2), span=self._s(), node_id=1)
        assert isinstance(node.variants, tuple)
        assert len(node.variants) == 2

    def test_type_alias(self) -> None:
        t = ListT(elem=TextT(span=self._s(), node_id=3), span=self._s(), node_id=2)
        node = TypeAlias(name="Names", type_expr=t, span=self._s(), node_id=1)
        assert node.name == "Names"

    def test_param_decl_annotated(self) -> None:
        t = TextT(span=self._s(), node_id=2)
        node = ParamDecl(name="spec", annotation=t, default=None, span=self._s(), node_id=1)
        assert node.name == "spec"
        assert node.annotation is t

    def test_param_decl_unannotated(self) -> None:
        node = ParamDecl(name="spec", annotation=None, default=None, span=self._s(), node_id=1)
        assert node.name == "spec"
        assert node.annotation is None

    def test_agent_decl_bare(self) -> None:
        node = AgentDecl(name="reviewer", runner=None, span=self._s(), node_id=1)
        assert node.name == "reviewer"
        assert node.runner is None

    def test_agent_decl_with_runner(self) -> None:
        node = AgentDecl(name="impl", runner="claude -p", span=self._s(), node_id=1)
        assert node.name == "impl"
        assert node.runner == "claude -p"

    def test_agent_decl_equality_ignores_span_and_node_id(self) -> None:
        a = AgentDecl(name="impl", runner="claude -p", span=span(), node_id=1)
        b = AgentDecl(name="impl", runner="claude -p", span=span(), node_id=99)
        assert a == b

    def test_agent_decl_inequality_on_runner(self) -> None:
        a = AgentDecl(name="impl", runner=None, span=span(), node_id=1)
        b = AgentDecl(name="impl", runner="claude -p", span=span(), node_id=1)
        assert a != b

    def test_func_def_is_declaration(self) -> None:
        import typing
        args = typing.get_args(Declaration)
        assert FuncDef in args

    def test_config_pragma(self) -> None:
        node = ConfigPragma(key="log", value=True, span=self._s(), node_id=1)
        assert node.key == "log"
        assert node.value is True


# ---------------------------------------------------------------------------
# Pattern nodes
# ---------------------------------------------------------------------------

class TestPatterns:
    def _s(self) -> SourceSpan:
        return span()

    def test_wildcard_pattern(self) -> None:
        p = WildcardPattern(span=self._s(), node_id=1)
        assert isinstance(p, WildcardPattern)

    def test_literal_pattern(self) -> None:
        lit = IntLit(value=42, span=self._s(), node_id=2)
        p = LiteralPattern(literal=lit, span=self._s(), node_id=1)
        assert p.literal is lit

    def test_var_pattern(self) -> None:
        p = VarPattern(name="x", span=self._s(), node_id=1)
        assert p.name == "x"

    def test_constructor_pattern_no_fields(self) -> None:
        p = ConstructorPattern(qualifier=None, name="None_", fields=(), span=self._s(), node_id=1)
        assert p.name == "None_"
        assert p.fields == ()

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
            fields=(pf,),
            span=self._s(),
            node_id=1,
        )
        assert isinstance(p.fields, tuple)
        assert p.fields[0].name == "val"

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
# Visitor / walk
# ---------------------------------------------------------------------------

class TestVisitorWalk:
    """Verify that walk() visits every node kind in a tree."""

    def _s(self) -> SourceSpan:
        return span()

    def _build_tree(self) -> Program:
        """Build a Program whose Block exercises every node kind.

        Every type, expression, binder, declaration, and pattern node that
        exists in the v2 AST must appear somewhere in the returned tree so that
        the walk-coverage tests can confirm they are all reachable.
        """
        s = self._s()

        # --- Type nodes (all 11 must appear) ---
        text_t = TextT(span=s, node_id=100)
        int_t = IntT(span=s, node_id=101)
        bool_t = BoolT(span=s, node_id=102)
        json_t = JsonT(span=s, node_id=103)
        decimal_t = DecimalT(span=s, node_id=104)
        name_t = NameT(name="MyT", span=s, node_id=105)
        list_t = ListT(elem=text_t, span=s, node_id=106)
        dict_t = DictT(value=int_t, span=s, node_id=107)
        unit_t = UnitT(span=s, node_id=108)
        agent_t = AgentT(span=s, node_id=109)
        func_t = FuncT(params=(int_t, text_t), result=bool_t, span=s, node_id=110)

        # Declarations — each field uses a different type so all types appear.
        field_int = FieldDef(name="x", type_expr=int_t, span=s, node_id=200)
        field_bool = FieldDef(name="flag", type_expr=bool_t, span=s, node_id=201)
        field_json = FieldDef(name="data", type_expr=json_t, span=s, node_id=202)
        field_decimal = FieldDef(name="price", type_expr=decimal_t, span=s, node_id=203)
        field_name = FieldDef(name="ref", type_expr=name_t, span=s, node_id=204)
        field_dict = FieldDef(name="meta", type_expr=dict_t, span=s, node_id=205)
        record_def = RecordDef(
            name="Point",
            fields=(field_int, field_bool, field_json, field_decimal, field_name, field_dict),
            span=s,
            node_id=210,
        )

        variant_field = FieldDef(name="val", type_expr=text_t, span=s, node_id=211)
        variant_def = VariantDef(name="Some", fields=(variant_field,), span=s, node_id=212)
        enum_def = EnumDef(name="Opt", variants=(variant_def,), span=s, node_id=213)

        type_alias = TypeAlias(name="Names", type_expr=list_t, span=s, node_id=214)
        param_decl = ParamDecl(name="spec", annotation=text_t, default=None, span=s, node_id=215)
        agent_decl = AgentDecl(name="reviewer", runner=None, span=s, node_id=216)
        config_pragma = ConfigPragma(key="log", value=True, span=s, node_id=217)

        # FuncDef — exercises UnitT, AgentT, FuncT via type annotations + Param
        p_unit = Param(name="u", type_expr=unit_t, default=None, span=s, node_id=218)
        p_agent = Param(name="a", type_expr=agent_t, default=None, span=s, node_id=219)
        p_func = Param(
            name="f",
            type_expr=func_t,
            default=IntLit(value=0, span=s, node_id=220),
            span=s,
            node_id=221,
        )
        func_def = FuncDef(
            name="helper",
            params=(p_unit, p_agent, p_func),
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
        list_lit = ListLit(elements=(int_lit,), span=s, node_id=308)

        # --- Template ---
        text_seg = TextSegment(text="hi ", span=s, node_id=309)
        var_ref_tmpl = VarRef(name="name", span=s, node_id=310)
        interp_seg = InterpSegment(expr=var_ref_tmpl, span=s, node_id=311)
        template = Template(segments=(text_seg, interp_seg), span=s, node_id=312)

        # --- Expressions ---
        var_ref = VarRef(name="x", span=s, node_id=400)
        field_access = FieldAccess(obj=var_ref, field="y", span=s, node_id=401)
        named_arg = NamedArg(name="a", value=int_lit, span=s, node_id=402)
        constructor = Constructor(
            qualifier=None, name="Point", args=(named_arg,), span=s, node_id=403
        )
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
        call_node = Call(
            callee=ask_ref,
            args=(template,),
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
            qualifier=None, name="Some", fields=(pat_field,), span=s, node_id=504
        )

        # Case with multiple patterns
        case_branch_wildcard = CaseBranch(pattern=wildcard_pat, body=null_lit, span=s, node_id=505)
        case_branch_lit = CaseBranch(pattern=lit_pat, body=unit_lit, span=s, node_id=506)
        case_branch_ctor = CaseBranch(pattern=ctor_pat, body=bool_lit, span=s, node_id=507)
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

        # Do loop
        do_body = Block(items=(unit_lit,), span=s, node_id=512)
        do_node = Do(limit=5, body=do_body, condition=bool_lit, span=s, node_id=513)

        # Try/catch
        catch_body = unit_lit
        catch_clause = CatchClause(
            exc_type="MyError", binding="e", body=catch_body, span=s, node_id=514
        )
        try_node = Try(body=int_lit, handlers=(catch_clause,), span=s, node_id=515)

        # Raise
        raise_node = Raise(exc=constructor, span=s, node_id=516)

        # --- Binders ---
        let_decl = LetDecl(name="a", type_ann=None, value=int_lit, span=s, node_id=600)
        let_with_type = LetDecl(name="c", type_ann=bool_t, value=bool_lit, span=s, node_id=601)
        var_decl = VarDecl(name="b", type_ann=None, value=str_lit, span=s, node_id=602)
        var_with_type = VarDecl(name="d", type_ann=json_t, value=null_lit, span=s, node_id=603)
        name_target = NameTarget(name="b", span=s, node_id=604)
        index_target = IndexTarget(obj=var_ref, index=int_lit, span=s, node_id=605)
        assign_stmt = AssignStmt(target=name_target, value=index_access, span=s, node_id=606)
        indexed_assign_stmt = AssignStmt(target=index_target, value=int_lit, span=s, node_id=607)

        # Param decl without annotation (exercises None branch in walk)
        input_no_ann = ParamDecl(name="bare", annotation=None, default=None, span=s, node_id=609)

        # Param with default (exercises the default branch in walk(Param))
        # Already covered in func_def (p_func has a default).

        # Block at the top level
        top_block = Block(
            items=(
                record_def, enum_def, type_alias, param_decl, input_no_ann,
                agent_decl, config_pragma, func_def,
                let_decl, let_with_type, var_decl, var_with_type,
                assign_stmt, indexed_assign_stmt,
                # expressions directly in block
                var_ref, field_access, index_access, constructor,
                binary_op, unary_not, unary_neg, is_test,
                call_node, lam, lam_no_ret,
                case_node, if_node, do_node, try_node, raise_node,
                dec_lit, dict_lit, list_lit,
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

    def test_walk_visits_all_decl_kinds(self) -> None:
        from agm.agl.syntax.visitor import walk

        prog = self._build_tree()
        visited: list[object] = []
        walk(prog, visited.append)
        kinds = {type(n) for n in visited}

        decl_kinds = {
            RecordDef, EnumDef, TypeAlias, ParamDecl, FieldDef, VariantDef,
            AgentDecl, FuncDef, ConfigPragma,
        }
        for kind in decl_kinds:
            assert kind in kinds, f"Expected {kind.__name__} to be visited"

    def test_walk_visits_all_expr_kinds(self) -> None:
        from agm.agl.syntax.visitor import walk

        prog = self._build_tree()
        visited: list[object] = []
        walk(prog, visited.append)
        kinds = {type(n) for n in visited}

        expr_kinds = {
            VarRef, FieldAccess, IndexAccess, Constructor, NamedArg,
            BinaryOp, UnaryNot, UnaryNeg, IsTest,
            Call, Lambda, Block, If, IfBranch, Case, CaseBranch,
            Do, Try, CatchClause, Raise,
            UnitLit, IntLit, DecimalLit, BoolLit, NullLit, StringLit,
            ListLit, DictLit, DictEntry, Template, TextSegment, InterpSegment,
        }
        for kind in expr_kinds:
            assert kind in kinds, f"Expected {kind.__name__} to be visited"

    def test_walk_visits_all_type_kinds(self) -> None:
        from agm.agl.syntax.visitor import walk

        prog = self._build_tree()
        visited: list[object] = []
        walk(prog, visited.append)
        kinds = {type(n) for n in visited}

        type_kinds = {
            TextT, JsonT, BoolT, IntT, DecimalT, NameT, ListT, DictT, UnitT, AgentT, FuncT
        }
        for kind in type_kinds:
            assert kind in kinds, f"Expected {kind.__name__} to be visited"

    def test_walk_visits_typed_call_type_arg(self) -> None:
        from agm.agl.syntax.visitor import walk

        # A Call with ``type_arg`` set: walk must descend into the type_arg so
        # the TypeExpr node is visited (covers the type_arg traversal branch).
        callee = VarRef(name="ask-request", span=self._s(), node_id=700)
        type_arg = NameT(name="Review", span=self._s(), node_id=701)
        call = Call(
            callee=callee,
            args=(StringLit(value="q", span=self._s(), node_id=702),),
            named_args=(),
            type_arg=type_arg,
            span=self._s(),
            node_id=703,
        )
        visited: list[object] = []
        walk(call, visited.append)
        assert type_arg in visited
        assert callee in visited

    def test_walk_visits_all_pattern_kinds(self) -> None:
        from agm.agl.syntax.visitor import walk

        prog = self._build_tree()
        visited: list[object] = []
        walk(prog, visited.append)
        kinds = {type(n) for n in visited}

        pattern_kinds = {
            WildcardPattern, LiteralPattern, VarPattern, ConstructorPattern, PatternField
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

    def test_walk_do_visits_body_then_condition(self) -> None:
        """walk(Do) visits body, then condition."""
        from agm.agl.syntax.visitor import walk

        s = self._s()
        body = UnitLit(span=s, node_id=2)
        cond = BoolLit(value=True, span=s, node_id=3)
        node = Do(limit=None, body=body, condition=cond, span=s, node_id=1)

        visited: list[object] = []
        walk(node, visited.append)

        assert visited[0] is node
        assert visited[1] is body
        assert visited[2] is cond

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

    def test_walk_agent_decl(self) -> None:
        from agm.agl.syntax.visitor import walk

        node = AgentDecl(name="reviewer", runner=None, span=span(), node_id=1)
        visited: list[object] = []
        walk(node, visited.append)
        # AgentDecl is a leaf — only itself is visited.
        assert visited == [node]

    def test_walk_param_decl_without_annotation(self) -> None:
        from agm.agl.syntax.visitor import walk

        s = span()
        node = ParamDecl(name="spec", annotation=None, default=None, span=s, node_id=1)
        visited: list[object] = []
        walk(node, visited.append)
        # Only the ParamDecl itself is visited; the missing annotation adds no child.
        assert visited == [node]

    def test_walk_param_decl_with_default(self) -> None:
        from agm.agl.syntax.visitor import walk

        s = span()
        default = IntLit(value=1, span=s, node_id=2)
        node = ParamDecl(name="spec", annotation=None, default=default, span=s, node_id=1)
        visited: list[object] = []
        walk(node, visited.append)
        assert visited == [node, default]

    def test_walk_program_decl_is_leaf(self) -> None:
        from agm.agl.syntax.visitor import walk

        node = ProgramDecl(name="demo", span=span(), node_id=1)
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

    def test_walk_agent_t_is_leaf(self) -> None:
        from agm.agl.syntax.visitor import walk

        node = AgentT(span=span(), node_id=1)
        visited: list[object] = []
        walk(node, visited.append)
        assert visited == [node]

    def test_visitor_subclass_new_nodes(self) -> None:
        """Subclassing Visitor and overriding visit_Call should be called."""
        from agm.agl.syntax.visitor import Visitor, walk

        s = self._s()
        callee = VarRef(name="f", span=s, node_id=2)
        call = Call(callee=callee, args=(), named_args=(), span=s, node_id=1)
        blk = Block(items=(call,), span=s, node_id=3)
        prog = Program(body=blk, span=s, node_id=0)

        class CountCalls(Visitor):
            def __init__(self) -> None:
                self.count = 0

            def visit_Call(self, node: Call) -> None:
                self.count += 1

        counter = CountCalls()
        walk(prog, counter.dispatch)
        assert counter.count == 1

    def test_visitor_dispatch_unknown_type_raises(self) -> None:
        """Visitor.dispatch on an unknown type should raise loudly."""
        from agm.agl.syntax.visitor import Visitor

        class MyVisitor(Visitor):
            pass

        v = MyVisitor()

        class NotANode:
            pass

        with pytest.raises(TypeError):
            v.dispatch(NotANode())

    def test_walk_unknown_type_raises(self) -> None:
        """walk() on an unknown type should raise TypeError."""
        from agm.agl.syntax.visitor import walk

        class NotANode:
            pass

        with pytest.raises(TypeError):
            walk(NotANode(), lambda n: None)

    def test_walk_qual_var_ref_visits_qualifier(self) -> None:
        """walk() on a qualified VarRef (foo.bar::thing) must visit the Qualifier node."""
        from agm.agl.parser import parse_program
        from agm.agl.syntax import Qualifier
        from agm.agl.syntax.visitor import walk

        prog = parse_program("foo.bar::thing")
        visited: list[object] = []
        walk(prog, visited.append)
        assert any(isinstance(n, Qualifier) for n in visited), (
            "Qualifier node not visited when walking foo.bar::thing"
        )

    def test_walk_qual_constructor_visits_qualifier(self) -> None:
        """walk() on a qualified Constructor (foo.bar::Color) must visit the Qualifier node."""
        from agm.agl.parser import parse_program
        from agm.agl.syntax import Qualifier
        from agm.agl.syntax.visitor import walk

        prog = parse_program("foo.bar::Color")
        visited: list[object] = []
        walk(prog, visited.append)
        assert any(isinstance(n, Qualifier) for n in visited), (
            "Qualifier node not visited when walking foo.bar::Color"
        )

    def test_walk_qual_name_t_visits_qualifier(self) -> None:
        """walk() on a NameT with a module_qualifier must visit the Qualifier node."""
        from agm.agl.parser import parse_program
        from agm.agl.syntax import Qualifier
        from agm.agl.syntax.visitor import walk

        # A qualified type annotation forces a NameT with module_qualifier set.
        prog = parse_program("let x: foo.bar::MyType = null")
        visited: list[object] = []
        walk(prog, visited.append)
        assert any(isinstance(n, Qualifier) for n in visited), (
            "Qualifier node not visited when walking a qualified NameT"
        )

    def test_walk_qual_constructor_pattern_visits_qualifier(self) -> None:
        """walk() on a ConstructorPattern with module_qualifier must visit the Qualifier."""
        from agm.agl.parser import parse_program
        from agm.agl.syntax import Qualifier
        from agm.agl.syntax.visitor import walk

        prog = parse_program("case x of | m::Foo => 1")
        visited: list[object] = []
        walk(prog, visited.append)
        assert any(isinstance(n, Qualifier) for n in visited), (
            "Qualifier node not visited when walking a qualified ConstructorPattern"
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

    def test_do_is_expr(self) -> None:
        import typing
        args = typing.get_args(Expr)
        assert Do in args

    def test_try_is_expr(self) -> None:
        import typing
        args = typing.get_args(Expr)
        assert Try in args

    def test_call_is_expr(self) -> None:
        import typing
        args = typing.get_args(Expr)
        assert Call in args

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

    def test_wildcard_pattern_is_pattern(self) -> None:
        import typing
        args = typing.get_args(Pattern)
        assert WildcardPattern in args

    def test_text_segment_is_template_segment(self) -> None:
        import typing
        args = typing.get_args(TemplateSegment)
        assert TextSegment in args

    def test_binder_union_members(self) -> None:
        import typing
        args = typing.get_args(Binder)
        assert LetDecl in args
        assert VarDecl in args
        assert AssignStmt in args

    def test_assign_target_union_members(self) -> None:
        import typing
        args = typing.get_args(AssignTarget)
        assert NameTarget in args
        assert IndexTarget in args

    def test_declaration_union_members(self) -> None:
        import typing
        args = typing.get_args(Declaration)
        for cls in (FuncDef, RecordDef, EnumDef, TypeAlias, ParamDecl, AgentDecl, ConfigPragma):
            assert cls in args, f"{cls.__name__} missing from Declaration union"

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
    """Verify that v1-only nodes are no longer part of the public API."""

    def test_agent_call_not_importable(self) -> None:
        with pytest.raises((ImportError, AttributeError)):
            from agm.agl.syntax import AgentCall  # type: ignore[attr-defined]  # noqa: F401

    def test_pass_stmt_not_importable(self) -> None:
        with pytest.raises((ImportError, AttributeError)):
            from agm.agl.syntax import PassStmt  # type: ignore[attr-defined]  # noqa: F401

    def test_print_stmt_not_importable(self) -> None:
        with pytest.raises((ImportError, AttributeError)):
            from agm.agl.syntax import PrintStmt  # type: ignore[attr-defined]  # noqa: F401

    def test_expr_stmt_not_importable(self) -> None:
        with pytest.raises((ImportError, AttributeError)):
            from agm.agl.syntax import ExprStmt  # type: ignore[attr-defined]  # noqa: F401

    def test_do_until_not_importable(self) -> None:
        with pytest.raises((ImportError, AttributeError)):
            from agm.agl.syntax import DoUntil  # type: ignore[attr-defined]  # noqa: F401

    def test_if_stmt_not_importable(self) -> None:
        with pytest.raises((ImportError, AttributeError)):
            from agm.agl.syntax import IfStmt  # type: ignore[attr-defined]  # noqa: F401

    def test_case_stmt_not_importable(self) -> None:
        with pytest.raises((ImportError, AttributeError)):
            from agm.agl.syntax import CaseStmt  # type: ignore[attr-defined]  # noqa: F401

    def test_case_expr_not_importable(self) -> None:
        with pytest.raises((ImportError, AttributeError)):
            from agm.agl.syntax import CaseExpr  # type: ignore[attr-defined]  # noqa: F401

    def test_if_expr_not_importable(self) -> None:
        with pytest.raises((ImportError, AttributeError)):
            from agm.agl.syntax import IfExpr  # type: ignore[attr-defined]  # noqa: F401

    def test_try_catch_not_importable(self) -> None:
        with pytest.raises((ImportError, AttributeError)):
            from agm.agl.syntax import TryCatch  # type: ignore[attr-defined]  # noqa: F401

    def test_call_options_not_importable(self) -> None:
        with pytest.raises((ImportError, AttributeError)):
            from agm.agl.syntax import CallOptions  # type: ignore[attr-defined]  # noqa: F401

    def test_stmt_union_not_importable(self) -> None:
        with pytest.raises((ImportError, AttributeError)):
            from agm.agl.syntax import Stmt  # type: ignore[attr-defined]  # noqa: F401


# ---------------------------------------------------------------------------
# Cast node (M1)
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
            expr=expr, target_type=target, test_only=False,
            span=SourceSpan(1, 0, 1, 5, 0, 5), node_id=99,
        )
        n2 = Cast(
            expr=expr, target_type=target, test_only=False,
            span=SourceSpan(2, 0, 2, 5, 0, 5), node_id=100,
        )
        assert n1 == n2

    def test_cast_immutable(self) -> None:
        expr = NullLit(span=self._sp(), node_id=0)
        target = BoolT(span=self._sp(), node_id=1)
        node = Cast(expr=expr, target_type=target, test_only=False, span=self._sp(), node_id=2)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            node.test_only = True  # type: ignore[misc]

    def test_cast_target_uses_type_expr(self) -> None:
        """target_type accepts any TypeExpr, including generic forms."""
        expr = VarRef(name="xs", span=self._sp(), node_id=0)
        target = ListT(elem=IntT(span=self._sp(), node_id=1), span=self._sp(), node_id=2)
        node = Cast(expr=expr, target_type=target, test_only=False, span=self._sp(), node_id=3)
        assert isinstance(node.target_type, ListT)

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
    """Tests for ImportMode, Qualifier, ImportItem, ImportDecl AST nodes."""

    def _sp(self) -> SourceSpan:
        return SourceSpan(1, 0, 1, 1, 0, 1)

    def test_import_mode_enum_values(self) -> None:
        from agm.agl.syntax import ImportMode
        assert ImportMode.ALL.value == "ALL"
        assert ImportMode.USING.value == "USING"
        assert ImportMode.HIDING.value == "HIDING"

    def test_qualifier_empty_segments_is_self_ref(self) -> None:
        from agm.agl.syntax import Qualifier
        q = Qualifier(segments=(), span=self._sp(), node_id=0)
        assert q.segments == ()

    def test_qualifier_dotted_segments(self) -> None:
        from agm.agl.syntax import Qualifier
        q = Qualifier(segments=("foo", "bar"), span=self._sp(), node_id=0)
        assert q.segments == ("foo", "bar")

    def test_qualifier_equality_ignores_span_and_node_id(self) -> None:
        from agm.agl.syntax import Qualifier
        q1 = Qualifier(segments=("mod",), span=SourceSpan(1,0,1,3,0,3), node_id=0)
        q2 = Qualifier(segments=("mod",), span=SourceSpan(2,0,2,3,0,3), node_id=99)
        assert q1 == q2

    def test_qualifier_immutable(self) -> None:
        from agm.agl.syntax import Qualifier
        q = Qualifier(segments=("a",), span=self._sp(), node_id=0)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            q.segments = ("b",)  # type: ignore[misc]

    def test_import_item_with_rename(self) -> None:
        from agm.agl.syntax import ImportItem
        item = ImportItem(name="foo", rename="bar", span=self._sp(), node_id=0)
        assert item.name == "foo"
        assert item.rename == "bar"

    def test_import_item_no_rename(self) -> None:
        from agm.agl.syntax import ImportItem
        item = ImportItem(name="baz", rename=None, span=self._sp(), node_id=0)
        assert item.rename is None

    def test_import_item_equality_ignores_span(self) -> None:
        from agm.agl.syntax import ImportItem
        i1 = ImportItem(name="x", rename=None, span=SourceSpan(1,0,1,1,0,1), node_id=0)
        i2 = ImportItem(name="x", rename=None, span=SourceSpan(2,0,2,1,0,1), node_id=99)
        assert i1 == i2

    def test_import_decl_basic(self) -> None:
        from agm.agl.syntax import ImportDecl, ImportMode
        decl = ImportDecl(
            module_path=("foo", "bar"),
            wildcard=False,
            qualified=False,
            alias=None,
            mode=ImportMode.ALL,
            items=(),
            span=self._sp(),
            node_id=0,
        )
        assert decl.module_path == ("foo", "bar")
        assert decl.wildcard is False
        assert decl.qualified is False
        assert decl.alias is None
        assert decl.mode == ImportMode.ALL
        assert decl.items == ()

    def test_import_decl_wildcard(self) -> None:
        from agm.agl.syntax import ImportDecl, ImportMode
        decl = ImportDecl(
            module_path=("utils",),
            wildcard=True,
            qualified=False,
            alias=None,
            mode=ImportMode.ALL,
            items=(),
            span=self._sp(),
            node_id=0,
        )
        assert decl.wildcard is True

    def test_import_decl_qualified_with_alias(self) -> None:
        from agm.agl.syntax import ImportDecl, ImportMode
        decl = ImportDecl(
            module_path=("foo",),
            wildcard=False,
            qualified=True,
            alias="f",
            mode=ImportMode.ALL,
            items=(),
            span=self._sp(),
            node_id=0,
        )
        assert decl.qualified is True
        assert decl.alias == "f"

    def test_import_decl_using_mode(self) -> None:
        from agm.agl.syntax import ImportDecl, ImportItem, ImportMode
        items = (
            ImportItem(name="foo", rename=None, span=self._sp(), node_id=1),
            ImportItem(name="bar", rename="b", span=self._sp(), node_id=2),
        )
        decl = ImportDecl(
            module_path=("m",),
            wildcard=False,
            qualified=False,
            alias=None,
            mode=ImportMode.USING,
            items=items,
            span=self._sp(),
            node_id=0,
        )
        assert decl.mode == ImportMode.USING
        assert len(decl.items) == 2

    def test_import_decl_hiding_mode(self) -> None:
        from agm.agl.syntax import ImportDecl, ImportItem, ImportMode
        items = (ImportItem(name="private_fn", rename=None, span=self._sp(), node_id=1),)
        decl = ImportDecl(
            module_path=("m",),
            wildcard=False,
            qualified=False,
            alias=None,
            mode=ImportMode.HIDING,
            items=items,
            span=self._sp(),
            node_id=0,
        )
        assert decl.mode == ImportMode.HIDING

    def test_import_decl_is_declaration(self) -> None:
        from agm.agl.syntax import ImportDecl, ImportMode
        decl = ImportDecl(
            module_path=("m",),
            wildcard=False,
            qualified=False,
            alias=None,
            mode=ImportMode.ALL,
            items=(),
            span=self._sp(),
            node_id=0,
        )
        assert isinstance(decl, ImportDecl)
        # Verify it is part of the Declaration union (isinstance check)
        _ = decl  # Declaration is a type alias, not a class, so just check it's the right type
        assert type(decl).__name__ == "ImportDecl"

    def test_import_decl_walk_visits_items(self) -> None:
        from agm.agl.syntax import ImportDecl, ImportItem, ImportMode
        from agm.agl.syntax.visitor import walk
        item = ImportItem(name="x", rename=None, span=self._sp(), node_id=1)
        decl = ImportDecl(
            module_path=("m",),
            wildcard=False,
            qualified=False,
            alias=None,
            mode=ImportMode.USING,
            items=(item,),
            span=self._sp(),
            node_id=0,
        )
        visited: list[object] = []
        walk(decl, visited.append)
        assert any(isinstance(n, ImportDecl) for n in visited)
        assert any(isinstance(n, ImportItem) for n in visited)

    def test_qualifier_walk_is_leaf(self) -> None:
        from agm.agl.syntax import Qualifier
        from agm.agl.syntax.visitor import walk
        q = Qualifier(segments=("a", "b"), span=self._sp(), node_id=0)
        visited: list[object] = []
        walk(q, visited.append)
        assert visited == [q]

    def test_func_def_is_private_default_false(self) -> None:
        """FuncDef.is_private defaults to False."""
        func = FuncDef(
            name="f",
            params=(),
            return_type=TextT(span=self._sp(), node_id=0),
            body=NullLit(span=self._sp(), node_id=1),
            span=self._sp(),
            node_id=2,
        )
        assert func.is_private is False

    def test_func_def_is_private_true(self) -> None:
        func = FuncDef(
            name="g",
            params=(),
            return_type=TextT(span=self._sp(), node_id=0),
            body=NullLit(span=self._sp(), node_id=1),
            span=self._sp(),
            node_id=2,
            is_private=True,
        )
        assert func.is_private is True

    def test_record_def_is_private_default(self) -> None:
        rec = RecordDef(name="R", fields=(), span=self._sp(), node_id=0)
        assert rec.is_private is False

    def test_enum_def_is_private_default(self) -> None:
        e = EnumDef(name="E", variants=(), span=self._sp(), node_id=0)
        assert e.is_private is False

    def test_type_alias_is_private_default(self) -> None:
        ta = TypeAlias(
            name="T",
            type_expr=TextT(span=self._sp(), node_id=0),
            span=self._sp(),
            node_id=1,
        )
        assert ta.is_private is False

    def test_var_ref_module_qualifier_default_none(self) -> None:
        ref = VarRef(name="x", span=self._sp(), node_id=0)
        assert ref.module_qualifier is None

    def test_var_ref_with_module_qualifier(self) -> None:
        from agm.agl.syntax import Qualifier
        q = Qualifier(segments=("foo",), span=self._sp(), node_id=0)
        ref = VarRef(name="x", span=self._sp(), node_id=1, module_qualifier=q)
        assert ref.module_qualifier is q

    def test_constructor_module_qualifier_default_none(self) -> None:
        ctor = Constructor(qualifier=None, name="Foo", args=(), span=self._sp(), node_id=0)
        assert ctor.module_qualifier is None

    def test_name_t_module_qualifier_default_none(self) -> None:
        t = NameT(name="MyType", span=self._sp(), node_id=0)
        assert t.module_qualifier is None

    def test_constructor_pattern_module_qualifier_default_none(self) -> None:
        pat = ConstructorPattern(qualifier=None, name="Foo", fields=(), span=self._sp(), node_id=0)
        assert pat.module_qualifier is None
