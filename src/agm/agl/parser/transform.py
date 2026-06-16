"""AstBuilder: Lark tree → agm.agl.syntax.* (the parser firewall).

This is the ONLY module in the codebase that imports both ``lark`` and
``agm.agl.syntax``.  Everything downstream of this module depends only on
the ``agm.agl.syntax`` dataclasses — never on Lark.

Design
------
- ``AstBuilder(lark.Transformer)`` with ``@v_args(meta=True)`` at class level
  so every rule method receives ``meta`` as its first positional argument.
- Monotonic ``node_id`` counter assigned per-node in document order.
- ``SourceSpan`` is derived from ``meta.line`` / ``meta.column`` /
  ``meta.end_line`` / ``meta.end_column`` / ``meta.start_pos`` /
  ``meta.end_pos`` — all provided by Lark when ``propagate_positions=True``.
- Both ``span`` and ``node_id`` are keyword-only ``compare=False`` fields in
  every AST dataclass, so equality tests ignore them.

Span convention for tokens used as leaves
------------------------------------------
When a rule has a single Token child and no meta (e.g. ``var_ref``), the
token's own position fields are used directly.  Rule-level meta is preferred
because it covers the full span of multi-token productions.
"""

from __future__ import annotations

import decimal
from itertools import count
from typing import cast

from lark import Transformer, v_args
from lark.lexer import Token
from lark.tree import Meta

import agm.agl.syntax as syntax
from agm.agl.parser.errors import AglSyntaxError
from agm.agl.syntax.nodes import ELSE
from agm.agl.syntax.spans import SourceSpan
from agm.agl.syntax.types import (
    BoolT,
    DecimalT,
    DictT,
    IntT,
    JsonT,
    ListT,
    NameT,
    TextT,
    TypeExpr,
)

# Types used internally for constructor-arg assembly
_NamedArgList = list[syntax.NamedArg]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_Args = list[object]  # Rule children after transformation (tokens + AST nodes)


def _span_from_meta(meta: Meta) -> SourceSpan:
    """Build a SourceSpan from Lark tree Meta (propagate_positions=True)."""
    return SourceSpan(
        start_line=meta.line,
        start_col=meta.column,
        end_line=meta.end_line,
        end_col=meta.end_column,
        start_offset=meta.start_pos,
        end_offset=meta.end_pos,
    )


def _span_from_token(tok: Token) -> SourceSpan:
    """Build a SourceSpan from a Lark Token's position fields."""
    line = tok.line if tok.line is not None else 1
    col = tok.column if tok.column is not None else 1
    pos = tok.start_pos if tok.start_pos is not None else 0
    end_line = tok.end_line if tok.end_line is not None else line
    end_col = tok.end_column if tok.end_column is not None else col + len(str(tok))
    end_pos = tok.end_pos if tok.end_pos is not None else pos + len(str(tok))
    return SourceSpan(
        start_line=line,
        start_col=col,
        end_line=end_line,
        end_col=end_col,
        start_offset=pos,
        end_offset=end_pos,
    )


# ---------------------------------------------------------------------------
# AstBuilder
# ---------------------------------------------------------------------------


@v_args(meta=True)
class AstBuilder(Transformer):
    """Transforms a Lark parse tree into ``agm.agl.syntax`` dataclasses.

    The ``node_id`` counter is monotonically increasing and starts at
    ``start_id`` (default ``0``).  Each ``parse_program`` call creates a fresh
    builder, so node IDs within a single program are deterministic (assigned
    in tree-walk order — root first, depth-first left-to-right).  Incremental
    sessions seed ``start_id`` from a prior parse's ``next_node_id`` so ids
    stay globally unique across entries.
    """

    def __init__(self, *, start_id: int = 0) -> None:
        super().__init__()
        self._counter = count(start_id)
        # The next id the counter will hand out.  Tracked explicitly so callers
        # can read the first id NOT consumed after a transform (the seed for a
        # subsequent incremental parse) without having to assume the root node
        # holds the maximum id.  Seeded to ``start_id`` before any node is built.
        self._next_unused: int = start_id
        # node_ids of Constructor nodes that already had a constructor_payload
        # applied via ctor_applied.  Used to reject a second payload
        # (``Issue(a: 1)(b: 2)``) regardless of whether the first payload was
        # empty (``Empty()``).  Tracking applied state explicitly avoids the
        # ``len(args) == 0`` ambiguity between "no payload" and "empty payload".
        self._payload_applied: set[int] = set()

    def _next_id(self) -> int:
        nid = next(self._counter)
        self._next_unused = nid + 1
        return nid

    @property
    def next_node_id(self) -> int:
        """The first ``node_id`` NOT yet consumed by this builder.

        After ``transform`` this is the seed (``start_id``) for the next
        incremental parse so that node ids stay globally unique across entries.
        Equal to ``start_id`` when no node has been built.
        """
        return self._next_unused

    # ------------------------------------------------------------------
    # Program root
    # ------------------------------------------------------------------

    def start(self, meta: Meta, args: _Args) -> syntax.Program:
        (block,) = args
        assert isinstance(block, tuple)
        stmts: tuple[syntax.Stmt, ...] = block
        span = _span_from_meta(meta)
        return syntax.Program(body=stmts, span=span, node_id=self._next_id())

    def block_stmts(self, meta: Meta, args: _Args) -> tuple[syntax.Stmt, ...]:
        # args contains stmts interleaved with SEMICOLON tokens.
        # _NEWLINE is filtered by the leading underscore convention, but
        # SEMICOLON (uppercase, %declare'd) appears in the tree.  Drop tokens.
        return tuple(
            s
            for s in args
            if s is not None and not isinstance(s, Token)
            if isinstance(s, (
                syntax.RecordDef, syntax.EnumDef, syntax.TypeAlias,
                syntax.InputDecl, syntax.AgentDecl, syntax.LetDecl, syntax.VarDecl,
                syntax.SetStmt, syntax.PassStmt, syntax.PrintStmt,
                syntax.ExprStmt, syntax.Raise,
                syntax.DoUntil, syntax.IfStmt, syntax.CaseStmt, syntax.TryCatch,
            ))
        )

    # ------------------------------------------------------------------
    # closed_stmt is transparent (?-prefixed via stmt)
    # ------------------------------------------------------------------

    def closed_stmt(self, meta: Meta, args: _Args) -> syntax.Stmt:
        (inner,) = args
        assert isinstance(inner, (
            syntax.RecordDef, syntax.EnumDef, syntax.TypeAlias,
            syntax.InputDecl, syntax.AgentDecl, syntax.LetDecl, syntax.VarDecl,
            syntax.SetStmt, syntax.PassStmt, syntax.PrintStmt,
            syntax.ExprStmt, syntax.Raise, syntax.DoUntil,
        ))
        return inner

    # open_stmt is transparent: if/case/try already return the right type
    def open_stmt(self, meta: Meta, args: _Args) -> syntax.Stmt:
        (inner,) = args
        assert isinstance(inner, (syntax.IfStmt, syntax.CaseStmt, syntax.TryCatch))
        return inner

    # bar_closed_stmt: transparent — delegates to the non-bar sub-rule return values
    def bar_closed_stmt(self, meta: Meta, args: _Args) -> syntax.Stmt:
        (inner,) = args
        assert isinstance(inner, (
            syntax.TypeAlias, syntax.LetDecl, syntax.VarDecl,
            syntax.SetStmt, syntax.PassStmt, syntax.PrintStmt,
            syntax.ExprStmt, syntax.Raise, syntax.DoUntil,
        ))
        return inner

    # ------------------------------------------------------------------
    # input_decl
    # ------------------------------------------------------------------

    def input_decl(self, meta: Meta, args: _Args) -> syntax.InputDecl:
        name_tok = args[0]
        assert isinstance(name_tok, Token)
        # Optional type_ann: present as TypeExpr in args[1] when annotated, absent otherwise.
        ann: TypeExpr | None = _find_type_expr(args[1:]) if len(args) > 1 else None
        span = _span_from_meta(meta)
        return syntax.InputDecl(
            name=str(name_tok),
            annotation=ann,
            span=span,
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # agent_decl
    # ------------------------------------------------------------------

    def agent_decl(self, meta: Meta, args: _Args) -> syntax.AgentDecl:
        # Grammar: AGENT VAR_NAME (EQ template)?
        # The leading AGENT keyword token is kept in the tree (it is a declared
        # terminal, so Lark does not filter it); the agent name is the VAR_NAME.
        # The template transformer normalises a plain (non-interpolated) string
        # to a StringLit and keeps an interpolated one as a Template.
        name_tok = next(
            a for a in args if isinstance(a, Token) and a.type == "VAR_NAME"
        )
        runner_node = next(
            (a for a in args if isinstance(a, (syntax.StringLit, syntax.Template))),
            None,
        )
        runner: str | None = (
            None
            if runner_node is None
            else _require_literal_string(
                runner_node,
                "agent runner string must be a literal string with no "
                "interpolation.",
            ).value
        )
        span = _span_from_meta(meta)
        return syntax.AgentDecl(
            name=str(name_tok),
            runner=runner,
            span=span,
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # let_decl / var_decl / set_stmt  (and bar twins)
    # ------------------------------------------------------------------

    def let_decl(self, meta: Meta, args: _Args) -> syntax.LetDecl:
        # Grammar: "let" VAR_NAME type_ann? EQ expr
        name_tok = args[0]
        assert isinstance(name_tok, Token)
        ann, value = _extract_ann_and_value(args[1:])
        span = _span_from_meta(meta)
        return syntax.LetDecl(
            name=str(name_tok),
            type_ann=ann,
            value=value,
            span=span,
            node_id=self._next_id(),
        )

    def var_decl(self, meta: Meta, args: _Args) -> syntax.VarDecl:
        name_tok = args[0]
        assert isinstance(name_tok, Token)
        ann, value = _extract_ann_and_value(args[1:])
        span = _span_from_meta(meta)
        return syntax.VarDecl(
            name=str(name_tok),
            type_ann=ann,
            value=value,
            span=span,
            node_id=self._next_id(),
        )

    def set_stmt(self, meta: Meta, args: _Args) -> syntax.SetStmt:
        # Grammar: "set" VAR_NAME EQ expr
        name_tok = args[0]
        assert isinstance(name_tok, Token)
        value = _find_expr(args[1:])
        span = _span_from_meta(meta)
        return syntax.SetStmt(
            target=str(name_tok),
            value=value,
            span=span,
            node_id=self._next_id(),
        )

    # Bar-safe twins: identical to the non-bar forms because bar_expr aliases or_expr.
    def bar_let_decl(self, meta: Meta, args: _Args) -> syntax.LetDecl:
        return self.let_decl(meta, args)

    def bar_var_decl(self, meta: Meta, args: _Args) -> syntax.VarDecl:
        return self.var_decl(meta, args)

    def bar_set_stmt(self, meta: Meta, args: _Args) -> syntax.SetStmt:
        return self.set_stmt(meta, args)

    def bar_print_stmt(self, meta: Meta, args: _Args) -> syntax.PrintStmt:
        return self.print_stmt(meta, args)

    def bar_raise_stmt(self, meta: Meta, args: _Args) -> syntax.Raise:
        return self.raise_stmt(meta, args)

    # ------------------------------------------------------------------
    # M2: record_def / field_def
    # ------------------------------------------------------------------

    def record_def(self, meta: Meta, args: _Args) -> syntax.RecordDef:
        # Grammar: "record" TYPE_NAME _NEWLINE _INDENT field_def+ _DEDENT
        # args: [Token(TYPE_NAME), FieldDef, FieldDef, ...]
        name_tok = args[0]
        assert isinstance(name_tok, Token)
        fields = tuple(a for a in args[1:] if isinstance(a, syntax.FieldDef))
        return syntax.RecordDef(
            name=str(name_tok),
            fields=fields,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def field_def(self, meta: Meta, args: _Args) -> syntax.FieldDef:
        # Grammar: field_name COLON type_expr
        name_tok = args[0]
        assert isinstance(name_tok, Token)
        type_expr = _find_type_expr(args[1:])
        return syntax.FieldDef(
            name=str(name_tok),
            type_expr=type_expr,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # M2: enum_def / variant_def / variant_payload / field_list / field_inline
    # ------------------------------------------------------------------

    def enum_def(self, meta: Meta, args: _Args) -> syntax.EnumDef:
        # Grammar: "enum" TYPE_NAME _NEWLINE _INDENT variant_def+ _DEDENT
        name_tok = args[0]
        assert isinstance(name_tok, Token)
        variants = tuple(a for a in args[1:] if isinstance(a, syntax.VariantDef))
        return syntax.EnumDef(
            name=str(name_tok),
            variants=variants,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def variant_def(self, meta: Meta, args: _Args) -> syntax.VariantDef:
        # Grammar: PIPE TYPE_NAME variant_payload?
        # args: [Token(PIPE), Token(TYPE_NAME), fields_tuple_or_nothing]
        name_tok = next((a for a in args if isinstance(a, Token) and a.type == "TYPE_NAME"), None)
        assert name_tok is not None, "variant_def: no TYPE_NAME token"
        # variant_payload returns a tuple[FieldDef, ...]; may be absent.
        fields: tuple[syntax.FieldDef, ...] = ()
        for a in args:
            if isinstance(a, tuple):
                fields = cast(tuple[syntax.FieldDef, ...], a)
                break
        return syntax.VariantDef(
            name=str(name_tok),
            fields=fields,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def variant_payload(self, meta: Meta, args: _Args) -> tuple[syntax.FieldDef, ...]:
        # Grammar: LPAR field_list? RPAR
        # args: [Token(LPAR), fields_tuple_or_None, Token(RPAR)]
        # field_list returns a tuple[FieldDef, ...]; when absent (empty payload),
        # maybe_placeholders=True yields None.
        for a in args:
            if isinstance(a, tuple):
                return cast(tuple[syntax.FieldDef, ...], a)
        return ()

    def field_list(self, meta: Meta, args: _Args) -> tuple[syntax.FieldDef, ...]:
        # Grammar: field_inline (COMMA field_inline)* COMMA?
        return tuple(a for a in args if isinstance(a, syntax.FieldDef))

    def field_inline(self, meta: Meta, args: _Args) -> syntax.FieldDef:
        # Grammar: VAR_NAME COLON type_expr  (same as field_def)
        name_tok = args[0]
        assert isinstance(name_tok, Token)
        type_expr = _find_type_expr(args[1:])
        return syntax.FieldDef(
            name=str(name_tok),
            type_expr=type_expr,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # M2: type_alias
    # ------------------------------------------------------------------

    def type_alias(self, meta: Meta, args: _Args) -> syntax.TypeAlias:
        # Grammar: "type" TYPE_NAME EQ type_expr
        name_tok = next((a for a in args if isinstance(a, Token) and a.type == "TYPE_NAME"), None)
        assert name_tok is not None, "type_alias: no TYPE_NAME token"
        type_expr = _find_type_expr(args)
        return syntax.TypeAlias(
            name=str(name_tok),
            type_expr=type_expr,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # pass_stmt
    # ------------------------------------------------------------------

    def pass_stmt(self, meta: Meta, args: _Args) -> syntax.PassStmt:
        span = _span_from_meta(meta)
        return syntax.PassStmt(span=span, node_id=self._next_id())

    # ------------------------------------------------------------------
    # print_stmt
    # ------------------------------------------------------------------

    def print_stmt(self, meta: Meta, args: _Args) -> syntax.PrintStmt:
        value = _find_expr(args)
        span = _span_from_meta(meta)
        return syntax.PrintStmt(value=value, span=span, node_id=self._next_id())

    # ------------------------------------------------------------------
    # raise_stmt
    # ------------------------------------------------------------------

    def raise_stmt(self, meta: Meta, args: _Args) -> syntax.Raise:
        exc = _find_expr(args)
        return syntax.Raise(exc=exc, span=_span_from_meta(meta), node_id=self._next_id())

    # ------------------------------------------------------------------
    # expr_stmt
    # ------------------------------------------------------------------

    def expr_stmt(self, meta: Meta, args: _Args) -> syntax.ExprStmt:
        expr = _find_expr(args)
        span = _span_from_meta(meta)
        # Design constraint: bare equality expressions such as `n = 2` look like
        # accidental assignments (use `set n = 2` to mutate a variable).  Reject
        # them at the parse level with a helpful diagnostic.
        if (
            isinstance(expr, syntax.BinaryOp)
            and expr.op == syntax.BinOp.EQ
            and isinstance(expr.left, syntax.VarRef)
        ):
            raise AglSyntaxError(
                f"Bare assignment '{expr.left.name} = …' is not valid. "
                "Use 'set' to reassign a mutable variable.",
                span=span,
            )
        return syntax.ExprStmt(expr=expr, span=span, node_id=self._next_id())

    # ------------------------------------------------------------------
    # type_ann
    # ------------------------------------------------------------------

    def type_ann(self, meta: Meta, args: _Args) -> TypeExpr:
        # Grammar: type_ann: COLON type_expr
        # args = [Token(COLON, ':'), type_expr_node]
        # Find the TypeExpr (non-Token) child.
        return _find_type_expr(args)

    # ------------------------------------------------------------------
    # type_expr dispatch (grammar rule names)
    # ------------------------------------------------------------------

    def prim_type_or_name(self, meta: Meta, args: _Args) -> TypeExpr:
        """VAR_NAME used in type position — map to primitive or NameT."""
        tok = args[0]
        assert isinstance(tok, Token)
        name = str(tok)
        span = _span_from_meta(meta)
        nid = self._next_id()
        if name == "text":
            return TextT(span=span, node_id=nid)
        if name == "json":
            return JsonT(span=span, node_id=nid)
        if name == "bool":
            return BoolT(span=span, node_id=nid)
        if name == "int":
            return IntT(span=span, node_id=nid)
        if name == "decimal":
            return DecimalT(span=span, node_id=nid)
        # Anything else is a named type reference (e.g. a type alias VAR_NAME).
        return NameT(name=name, span=span, node_id=nid)

    def named_type(self, meta: Meta, args: _Args) -> NameT:
        """TYPE_NAME used in type position."""
        tok = args[0]
        assert isinstance(tok, Token)
        return NameT(name=str(tok), span=_span_from_meta(meta), node_id=self._next_id())

    def generic_type_1(self, meta: Meta, args: _Args) -> TypeExpr:
        """VAR_NAME LSQB type_expr RSQB — handles list[T]."""
        head_tok = args[0]
        assert isinstance(head_tok, Token)
        head = str(head_tok)
        inner: TypeExpr = _find_type_expr(args[1:])
        span = _span_from_meta(meta)
        nid = self._next_id()
        if head == "list":
            return ListT(elem=inner, span=span, node_id=nid)
        raise syntax_error_from_meta(meta, f"Unknown generic type: {head!r}")

    def dict_type(self, meta: Meta, args: _Args) -> DictT:
        """VAR_NAME LSQB VAR_NAME COMMA type_expr RSQB — dict[text, V].

        The head must be ``dict``; any other head is rejected.  In v1 the key
        type is always ``text``; any other key is rejected with the key token's
        span so the diagnostic pinpoints the offending type.
        """
        # args: [head VAR_NAME, LSQB, key VAR_NAME, COMMA, type_expr, RSQB].
        var_name_toks = [
            a for a in args if isinstance(a, Token) and a.type == "VAR_NAME"
        ]
        assert len(var_name_toks) >= 2, "dict_type: missing key token"
        head_tok = var_name_toks[0]
        if str(head_tok) != "dict":
            raise syntax_error_from_meta(meta, f"Unknown generic type: {str(head_tok)!r}")
        key_tok = var_name_toks[1]
        if str(key_tok) != "text":
            raise AglSyntaxError(
                f"dict keys are always text in v1, got {str(key_tok)!r}.",
                span=_span_from_token(key_tok),
            )
        value: TypeExpr = _find_type_expr(args[1:])
        return DictT(value=value, span=_span_from_meta(meta), node_id=self._next_id())

    # ------------------------------------------------------------------
    # paren_expr — transparent: just return the inner expr
    # ------------------------------------------------------------------

    def paren_expr(self, meta: Meta, args: _Args) -> syntax.Expr:
        # args: LPAR, expr, RPAR — find the expr (non-Token)
        for a in args:
            if not isinstance(a, Token) and a is not None:
                return cast(syntax.Expr, a)
        raise AssertionError("paren_expr: no inner expression found")

    # ------------------------------------------------------------------
    # Literals
    # ------------------------------------------------------------------

    def lit_int(self, meta: Meta, args: _Args) -> syntax.IntLit:
        tok = args[0]
        assert isinstance(tok, Token)
        return syntax.IntLit(
            value=int(str(tok)),
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def lit_decimal(self, meta: Meta, args: _Args) -> syntax.DecimalLit:
        tok = args[0]
        assert isinstance(tok, Token)
        return syntax.DecimalLit(
            value=decimal.Decimal(str(tok)),
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def lit_true(self, meta: Meta, args: _Args) -> syntax.BoolLit:
        return syntax.BoolLit(
            value=True, span=_span_from_meta(meta), node_id=self._next_id()
        )

    def lit_false(self, meta: Meta, args: _Args) -> syntax.BoolLit:
        return syntax.BoolLit(
            value=False, span=_span_from_meta(meta), node_id=self._next_id()
        )

    def lit_null(self, meta: Meta, args: _Args) -> syntax.NullLit:
        return syntax.NullLit(span=_span_from_meta(meta), node_id=self._next_id())

    # ------------------------------------------------------------------
    # var_ref
    # ------------------------------------------------------------------

    def var_ref(self, meta: Meta, args: _Args) -> syntax.VarRef:
        tok = args[0]
        assert isinstance(tok, Token)
        return syntax.VarRef(
            name=str(tok), span=_span_from_meta(meta), node_id=self._next_id()
        )

    # ------------------------------------------------------------------
    # M2: access-level rules (field_access, type_access, access transparent)
    # ------------------------------------------------------------------

    def access(self, meta: Meta, args: _Args) -> syntax.Expr:
        """access: atom — transparent; return the single Expr child.

        The ``access: atom`` production passes through a single non-Token child.
        """
        # args always contains exactly one non-Token element (the atom).
        expr = next(a for a in args if a is not None and not isinstance(a, Token))
        return cast(syntax.Expr, expr)

    def field_access(self, meta: Meta, args: _Args) -> syntax.FieldAccess:
        """access DOT field_name — record field access.

        ``field_name`` yields the field-name Token (type ``VAR_NAME`` or, for the
        reserved word ``agent``, ``AGENT``).  ``agent`` is reserved (it leads an
        agent declaration) but stays valid as a field name so built-in exception
        fields like ``AgentCallError.agent`` keep working.
        """
        obj_expr = cast(syntax.Expr, args[0])
        field_tok = _find_name_token(args)
        return syntax.FieldAccess(
            obj=obj_expr,
            field=str(field_tok),
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def type_access(self, meta: Meta, args: _Args) -> syntax.Constructor:
        """access DOT TYPE_NAME — qualified enum-variant constructor.

        Only ``TYPE_NAME . TYPE_NAME`` qualification is legal (design §10.13:
        ``qualified_constructor ::= TYPE_NAME | TYPE_NAME "." TYPE_NAME``).  The
        LHS must therefore be a bare, unqualified, no-arg ``Constructor``
        (a lone ``TYPE_NAME``).  Any other LHS is rejected:

        - ``x.Done``  — VAR_NAME / record / other expr on the LHS (F1).
        - ``A.B.C``   — the LHS is already a qualified Constructor (F2).
        - ``Empty(a: 1).Done`` — the LHS already carries constructor args.

        The error span pinpoints the offending ``.TypeName`` suffix.
        """
        obj_expr = args[0]
        type_name_tok = next(
            (a for a in args if isinstance(a, Token) and a.type == "TYPE_NAME"), None
        )
        assert type_name_tok is not None, "type_access: no TYPE_NAME token"
        variant_name = str(type_name_tok)
        if (
            isinstance(obj_expr, syntax.Constructor)
            and obj_expr.qualifier is None
            and not obj_expr.args
            and obj_expr.node_id not in self._payload_applied
        ):
            return syntax.Constructor(
                qualifier=obj_expr.name,
                name=variant_name,
                args=(),
                span=_span_from_meta(meta),
                node_id=self._next_id(),
            )
        raise AglSyntaxError(
            f"'.{variant_name}' may only follow a type name "
            "(qualified constructor); chained or non-type qualification is "
            "not allowed.",
            span=_span_from_token(type_name_tok),
        )

    def ctor_applied(self, meta: Meta, args: _Args) -> syntax.Constructor:
        """access constructor_payload — apply args to a bare constructor.

        Handles both unqualified ``Type(x: 1)`` and qualified ``Ns.Type(x: 1)``
        by taking the existing (possibly qualified) Constructor from the access
        position and attaching the named arguments from constructor_payload.

        ``args`` is always ``[<access>, list[NamedArg]]`` since
        ``constructor_payload`` is required (not optional) in this rule.

        Two forms are rejected as clean ``AglSyntaxError``s:

        - F4: the access base is not a ``Constructor`` (``x.f(arg)``,
          ``(x)(a: 1)``, ``1(a: 1)`` …).  Constructor arguments may only follow
          a type name; method-call syntax is not part of v1.
        - F3: the base ``Constructor`` already had a payload applied
          (``Issue(a: 1)(b: 2)``).  A constructor takes a single argument list.
          ``Status.Done`` followed by its *first* payload is still legal even
          when the type-access produced empty args.
        """
        ctor = args[0]
        payload = args[1]
        assert isinstance(payload, list), (
            f"ctor_applied: expected list payload, got {type(payload)}"
        )
        if not isinstance(ctor, syntax.Constructor):
            raise AglSyntaxError(
                "constructor arguments may only follow a type name; "
                "method-call syntax is not supported.",
                span=_span_from_meta(meta),
            )
        if ctor.node_id in self._payload_applied:
            # A second payload on an already-applied constructor.  Pin the error
            # to the second payload's span (everything after the base ctor).
            raise AglSyntaxError(
                "constructor takes a single argument list.",
                span=self._span_after(ctor.span, meta),
            )
        named_args = tuple(cast(_NamedArgList, payload))
        new_id = self._next_id()
        self._payload_applied.add(new_id)
        return syntax.Constructor(
            qualifier=ctor.qualifier,
            name=ctor.name,
            args=named_args,
            span=_span_from_meta(meta),
            node_id=new_id,
        )

    @staticmethod
    def _span_after(base: SourceSpan, meta: Meta) -> SourceSpan:
        """Span covering the payload that follows ``base`` within the rule meta.

        Used to pinpoint the *second* payload of ``Issue(a: 1)(b: 2)``: the
        offending ``(b: 2)`` starts where the base constructor's span ends.
        """
        return SourceSpan(
            start_line=base.end_line,
            start_col=base.end_col,
            end_line=meta.end_line,
            end_col=meta.end_column,
            start_offset=base.end_offset,
            end_offset=meta.end_pos,
        )

    # ------------------------------------------------------------------
    # M2: list literal
    # ------------------------------------------------------------------

    def lit_list(self, meta: Meta, args: _Args) -> syntax.ListLit:
        """lit_list: LSQB (expr (COMMA expr)* COMMA?)? RSQB"""
        elements = tuple(
            cast(syntax.Expr, a)
            for a in args
            if a is not None and not isinstance(a, Token)
        )
        return syntax.ListLit(
            elements=elements,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # M2: dict literal
    # ------------------------------------------------------------------

    def lit_dict(self, meta: Meta, args: _Args) -> syntax.DictLit:
        """lit_dict: LBRACE (dict_entry (COMMA dict_entry)* COMMA?)? RBRACE"""
        entries = tuple(a for a in args if isinstance(a, syntax.DictEntry))
        return syntax.DictLit(
            entries=entries,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def dict_entry_str(self, meta: Meta, args: _Args) -> syntax.DictEntry:
        """dict_entry: template COLON expr — quoted string key.

        The template transformer normalises plain strings to StringLit already.
        A plain ``StringLit`` key is valid (design §10.14: dict keys are
        ``STRING`` or ``VAR_NAME``).  A key carrying any interpolation segment
        (e.g. ``{"${a}": 1}``) is rejected — dict keys must be literal strings
        (F5).
        """
        non_tokens = [a for a in args if a is not None and not isinstance(a, Token)]
        assert len(non_tokens) >= 2, (
            f"dict_entry_str: expected key + expr, got {args!r}"
        )
        key_lit = _require_literal_string(
            non_tokens[0],
            "dict keys must be literal strings (no interpolation).",
        )
        val_expr = cast(syntax.Expr, non_tokens[1])
        return syntax.DictEntry(
            key=key_lit,
            value=val_expr,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def dict_entry_name(self, meta: Meta, args: _Args) -> syntax.DictEntry:
        """dict_entry: VAR_NAME COLON expr — identifier shorthand key.

        The identifier is converted to ``StringLit`` so the AST key is always
        ``StringLit``, matching ``DictEntry.key``.  ``field_name`` admits the
        reserved word ``agent`` as a shorthand key in addition to ordinary
        identifiers.
        """
        name_tok = _find_name_token(args)
        val_expr = _find_expr(args[1:])
        key_lit = syntax.StringLit(
            value=str(name_tok),
            span=_span_from_token(name_tok),
            node_id=self._next_id(),
        )
        return syntax.DictEntry(
            key=key_lit,
            value=val_expr,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # M2: constructor expressions
    # ------------------------------------------------------------------

    def ctor_unqualified(self, meta: Meta, args: _Args) -> syntax.Constructor:
        """constructor: TYPE_NAME → bare unqualified constructor (no args).

        The payload (if any) is applied by ``ctor_applied`` at the access level.
        """
        name_tok = next(
            (a for a in args if isinstance(a, Token) and a.type == "TYPE_NAME"), None
        )
        assert name_tok is not None, "ctor_unqualified: no TYPE_NAME token"
        return syntax.Constructor(
            qualifier=None,
            name=str(name_tok),
            args=(),
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def constructor_payload(self, meta: Meta, args: _Args) -> _NamedArgList:
        """constructor_payload: LPAR constructor_args? RPAR → list[NamedArg]."""
        for a in args:
            if isinstance(a, list):
                return cast(_NamedArgList, a)
        return []

    def constructor_args(self, meta: Meta, args: _Args) -> _NamedArgList:
        """constructor_args: named_arg (COMMA named_arg)* COMMA?

        Duplicate argument names are rejected with the span of the duplicate.
        """
        result: list[syntax.NamedArg] = []
        seen: dict[str, SourceSpan] = {}
        for a in args:
            if not isinstance(a, syntax.NamedArg):
                continue
            if a.name in seen:
                raise AglSyntaxError(
                    f"duplicate constructor argument {a.name!r}.",
                    span=a.span,
                )
            seen[a.name] = a.span
            result.append(a)
        return result

    def named_arg(self, meta: Meta, args: _Args) -> syntax.NamedArg:
        """named_arg: field_name COLON expr

        ``field_name`` admits the reserved word ``agent`` as an argument name so
        built-in exception constructors (e.g. ``AgentCallError(agent: …)``) can
        be written.
        """
        name_tok = _find_name_token(args)
        val_expr = _find_expr(args[1:])
        return syntax.NamedArg(
            name=str(name_tok),
            value=val_expr,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # agent_call
    # ------------------------------------------------------------------

    def agent_call(self, meta: Meta, args: _Args) -> syntax.AgentCall:
        # Grammar: VAR_NAME call_options? template
        # args when no options:   [Token(VAR_NAME), Template|StringLit]
        # args when with options: [Token(VAR_NAME), CallOptions, Template|StringLit]
        # Note: the template transformer normalises plain strings to StringLit;
        # AgentCall.template requires Template, so we wrap StringLit back here.
        name_tok = args[0]
        assert isinstance(name_tok, Token)

        options: syntax.CallOptions | None = None
        tmpl: syntax.Template | None = None

        for a in args[1:]:
            if isinstance(a, syntax.CallOptions):
                options = a
            if isinstance(a, syntax.Template):
                tmpl = a
            elif isinstance(a, syntax.StringLit):
                # Normalised from a plain-text template — re-wrap as Template.
                text_seg = syntax.TextSegment(
                    text=a.value, span=a.span, node_id=self._next_id()
                )
                tmpl = syntax.Template(
                    segments=(text_seg,),
                    span=a.span,
                    node_id=self._next_id(),
                )

        assert tmpl is not None, "agent_call: no Template found"

        if options is None:
            # No options specified — use an empty CallOptions.
            opt_span = _span_from_token(name_tok)
            options = syntax.CallOptions(
                format=None,
                strict_json=None,
                parse_policy=None,
                span=opt_span,
                node_id=self._next_id(),
            )

        span = _span_from_meta(meta)
        return syntax.AgentCall(
            agent=str(name_tok),
            options=options,
            template=tmpl,
            span=span,
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # call_options
    # ------------------------------------------------------------------

    def call_options(self, meta: Meta, args: _Args) -> syntax.CallOptions:
        fmt: str | None = None
        strict_json: bool | None = None
        parse_policy: syntax.ParsePolicy | None = None
        seen: set[str] = set()

        for opt in args:
            if isinstance(opt, (_FormatOpt, _StrictJsonOpt, _ParseErrorOpt)):
                if opt.key in seen:
                    # Reject the second occurrence of any option key, pinning the
                    # error to the duplicate's span (design §4.3).
                    raise AglSyntaxError(
                        f"duplicate option {opt.key!r}.", span=opt.span
                    )
                seen.add(opt.key)
            if isinstance(opt, _FormatOpt):
                fmt = opt.value
            elif isinstance(opt, _StrictJsonOpt):
                strict_json = opt.value
            elif isinstance(opt, _ParseErrorOpt):
                parse_policy = opt.value
            # Tokens (LSQB, RSQB, COMMA) are silently ignored.

        span = _span_from_meta(meta)
        return syntax.CallOptions(
            format=fmt,
            strict_json=strict_json,
            parse_policy=parse_policy,
            span=span,
            node_id=self._next_id(),
        )

    def opt_raw(
        self, meta: Meta, args: _Args
    ) -> "_FormatOpt | _StrictJsonOpt | _ParseErrorOpt":
        """call_option: VAR_NAME COLON call_option_value — dispatch on key name."""
        key_tok = args[0]
        assert isinstance(key_tok, Token)
        key = str(key_tok)
        # Last arg that is not a Token is the option value node.
        val_node = args[-1]
        span = _span_from_meta(meta)

        if key == "format":
            # format: text | json | VAR_NAME  (val_node is str from opt_val_name)
            if isinstance(val_node, _NameOrPolicy) and isinstance(val_node.value, str):
                return _FormatOpt(val_node.value, span)
            raise AglSyntaxError(
                f"format expects a format name (e.g. text, json), got {val_node!r}.",
                span=span,
            )

        if key == "strict_json":
            if isinstance(val_node, bool):
                return _StrictJsonOpt(val_node, span)
            raise AglSyntaxError(
                f"strict_json expects true or false, got {val_node!r}.",
                span=span,
            )

        if key == "on_parse_error":
            # val_node is _NameOrPolicy(AbortPolicy) or RetryPolicy
            if isinstance(val_node, _NameOrPolicy) and isinstance(
                val_node.value, (syntax.AbortPolicy, syntax.RetryPolicy)
            ):
                return _ParseErrorOpt(val_node.value, span)
            if isinstance(val_node, syntax.RetryPolicy):
                return _ParseErrorOpt(val_node, span)
            raise AglSyntaxError(
                f"on_parse_error expects abort or retry[N], got {val_node!r}.",
                span=span,
            )

        raise AglSyntaxError(f"Unknown call option: {key!r}.", span=span)

    def opt_val_true(self, meta: Meta, args: _Args) -> bool:
        return True

    def opt_val_false(self, meta: Meta, args: _Args) -> bool:
        return False

    def opt_val_name(self, meta: Meta, args: _Args) -> "_NameOrPolicy":
        tok = args[0]
        assert isinstance(tok, Token)
        name = str(tok)
        if name == "abort":
            return _NameOrPolicy(
                syntax.AbortPolicy(span=_span_from_meta(meta), node_id=self._next_id())
            )
        return _NameOrPolicy(name)

    def opt_val_name_int(self, meta: Meta, args: _Args) -> syntax.RetryPolicy:
        # VAR_NAME LSQB INT RSQB — must be retry[N]
        name_tok = args[0]
        assert isinstance(name_tok, Token)
        name = str(name_tok)
        if name != "retry":
            raise AglSyntaxError(
                f"Expected 'retry[N]', got {name!r}.",
                span=_span_from_meta(meta),
            )
        int_tok = next(
            (t for t in args if isinstance(t, Token) and t.type == "INT"), None
        )
        assert int_tok is not None, "opt_val_name_int: no INT token found"
        return syntax.RetryPolicy(
            extra=int(str(int_tok)),
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # M3: Binary operators
    # ------------------------------------------------------------------

    def _binary(self, meta: Meta, args: _Args, op: syntax.BinOp) -> syntax.BinaryOp:
        left = cast(syntax.Expr, args[0])
        right = cast(syntax.Expr, args[-1])
        return syntax.BinaryOp(
            op=op, left=left, right=right,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    def bin_or(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.OR)

    def bin_and(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.AND)

    def bin_eq(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.EQ)

    def bin_neq(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.NEQ)

    def bin_lt(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.LT)

    def bin_le(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.LE)

    def bin_gt(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.GT)

    def bin_ge(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.GE)

    def bin_in(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.IN)

    def bin_add(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.ADD)

    def bin_sub(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.SUB)

    def bin_mul(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.MUL)

    def bin_div(self, meta: Meta, args: _Args) -> syntax.BinaryOp:
        return self._binary(meta, args, syntax.BinOp.DIV)

    # ------------------------------------------------------------------
    # M3: Unary operators
    # ------------------------------------------------------------------

    def unary_not(self, meta: Meta, args: _Args) -> syntax.UnaryNot:
        operand = cast(syntax.Expr, args[0])
        return syntax.UnaryNot(
            operand=operand, span=_span_from_meta(meta), node_id=self._next_id()
        )

    def unary_neg(self, meta: Meta, args: _Args) -> syntax.UnaryNeg:
        # args: [Token(MINUS), expr]
        operand = cast(syntax.Expr, args[-1])
        return syntax.UnaryNeg(
            operand=operand, span=_span_from_meta(meta), node_id=self._next_id()
        )

    # ------------------------------------------------------------------
    # M3: is / is not tests
    # ------------------------------------------------------------------

    def is_test_simple(self, meta: Meta, args: _Args) -> syntax.IsTest:
        # additive "is" TYPE_NAME
        left = cast(syntax.Expr, args[0])
        variant_tok = next(a for a in args if isinstance(a, Token) and a.type == "TYPE_NAME")
        return syntax.IsTest(
            expr=left, qualifier=None, variant=str(variant_tok), negated=False,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    def is_test_qualified(self, meta: Meta, args: _Args) -> syntax.IsTest:
        # additive "is" TYPE_NAME DOT TYPE_NAME
        left = cast(syntax.Expr, args[0])
        type_toks = [a for a in args if isinstance(a, Token) and a.type == "TYPE_NAME"]
        assert len(type_toks) == 2
        return syntax.IsTest(
            expr=left, qualifier=str(type_toks[0]), variant=str(type_toks[1]), negated=False,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    def is_not_test_simple(self, meta: Meta, args: _Args) -> syntax.IsTest:
        # additive "is" "not" TYPE_NAME
        left = cast(syntax.Expr, args[0])
        variant_tok = next(a for a in args if isinstance(a, Token) and a.type == "TYPE_NAME")
        return syntax.IsTest(
            expr=left, qualifier=None, variant=str(variant_tok), negated=True,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    def is_not_test_qualified(self, meta: Meta, args: _Args) -> syntax.IsTest:
        # additive "is" "not" TYPE_NAME DOT TYPE_NAME
        left = cast(syntax.Expr, args[0])
        type_toks = [a for a in args if isinstance(a, Token) and a.type == "TYPE_NAME"]
        assert len(type_toks) == 2
        return syntax.IsTest(
            expr=left, qualifier=str(type_toks[0]), variant=str(type_toks[1]), negated=True,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # M3: case_expr
    # ------------------------------------------------------------------

    def case_expr(self, meta: Meta, args: _Args) -> syntax.CaseExpr:
        # args: [expr, CaseExprBranch, CaseExprBranch, ...]
        subject = cast(syntax.Expr, args[0])
        branches = tuple(a for a in args[1:] if isinstance(a, syntax.CaseExprBranch))
        return syntax.CaseExpr(
            subject=subject, branches=branches,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    def case_expr_branch(self, meta: Meta, args: _Args) -> syntax.CaseExprBranch:
        # args: [pattern, ARROW token, body_expr] — find the pattern and body
        pat_types = (
            syntax.WildcardPattern, syntax.LiteralPattern,
            syntax.VarPattern, syntax.ConstructorPattern,
        )
        pat = next(a for a in args if isinstance(a, pat_types))
        body = _find_expr([a for a in args if not isinstance(a, pat_types)])
        assert isinstance(pat, pat_types)
        return syntax.CaseExprBranch(
            pattern=pat, body=body,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # M3: do_until
    # ------------------------------------------------------------------

    def loop_bound(self, meta: Meta, args: _Args) -> int:
        """Extract N from a LOOP_BOUND token and validate it is positive."""
        tok = args[0]
        assert isinstance(tok, Token)
        n = int(str(tok))
        if n <= 0:
            raise AglSyntaxError(
                f"Loop bound must be a positive integer; got {n}.",
                span=_span_from_meta(meta),
            )
        return n

    def body(self, meta: Meta, args: _Args) -> tuple[syntax.Stmt, ...]:
        """Body of a do_until: either a suite or inline_seq."""
        (inner,) = args
        assert isinstance(inner, tuple)
        return cast(tuple[syntax.Stmt, ...], inner)

    def inline_seq(self, meta: Meta, args: _Args) -> tuple[syntax.Stmt, ...]:
        """inline_seq: closed_stmt (SEMICOLON closed_stmt)* (SEMICOLON do_tail)? | do_tail"""
        # Filter out only SEMICOLON tokens; all AST node types (closed_stmt and
        # do_tail results) are preserved so the scope resolver can reject type/input
        # declarations that appear at a non-root position.
        return tuple(
            cast(syntax.Stmt, s) for s in args if s is not None and not isinstance(s, Token)
        )

    def do_tail(self, meta: Meta, args: _Args) -> syntax.Stmt:
        """do_tail: if_stmt | case_stmt | try_stmt — transparent."""
        (inner,) = args
        assert isinstance(inner, (syntax.IfStmt, syntax.CaseStmt, syntax.TryCatch))
        return inner

    def suite(self, meta: Meta, args: _Args) -> tuple[syntax.Stmt, ...]:
        """suite: _INDENT block_stmts _DEDENT — unwrap the block_stmts tuple."""
        # block_stmts always returns a tuple[Stmt,...]; it's the only non-Token child.
        stmts = next(a for a in args if isinstance(a, tuple))
        return cast(tuple[syntax.Stmt, ...], stmts)

    def do_until(self, meta: Meta, args: _Args) -> syntax.DoUntil:
        # Grammar: "do" loop_bound? body "until" bar_expr
        # After transformation (with maybe_placeholders=True):
        #   bounded:   [int, tuple[Stmt,...], Expr]
        #   unbounded: [tuple[Stmt,...], Expr]
        # Anonymous terminals ("do", "until") are filtered by Lark.
        # loop_bound? absent: no None placeholder — the item is simply omitted.
        limit: int | None = None
        body_stmts: tuple[syntax.Stmt, ...] = ()
        cond: syntax.Expr | None = None
        for a in args:
            if isinstance(a, int):
                limit = a
            elif isinstance(a, tuple):
                body_stmts = cast(tuple[syntax.Stmt, ...], a)
            else:
                cond = cast(syntax.Expr, a)
        assert cond is not None, "do_until: no condition"
        return syntax.DoUntil(
            limit=limit, body=body_stmts, condition=cond,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # M3: if_stmt
    # ------------------------------------------------------------------

    def if_cond_branch(self, meta: Meta, args: _Args) -> syntax.IfBranch:
        # Grammar: bar_expr ARROW branch_body
        # After transformation: [Expr, tuple[Stmt,...]]
        cond = cast(syntax.Expr, args[0])
        body = _find_branch_body(args[1:])
        return syntax.IfBranch(
            cond=cond, body=body,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    def if_else_branch(self, meta: Meta, args: _Args) -> syntax.IfBranch:
        # Grammar: "else" ARROW branch_body
        body = _find_branch_body(args)
        return syntax.IfBranch(
            cond=ELSE, body=body,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    def if_stmt(self, meta: Meta, args: _Args) -> syntax.IfStmt:
        branches = tuple(a for a in args if isinstance(a, syntax.IfBranch))
        # Validate: else must be last
        for i, b in enumerate(branches):
            if b.cond is ELSE and i < len(branches) - 1:
                raise AglSyntaxError(
                    "'else' branch must be the last branch in an if statement.",
                    span=b.span,
                )
        return syntax.IfStmt(
            branches=branches,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    def branch_body(self, meta: Meta, args: _Args) -> tuple[syntax.Stmt, ...]:
        """branch_body: suite | bar_closed_stmt | try_stmt

        Returns a flat tuple of statements.  The grammar guarantees exactly one
        non-None, non-Token child:
          * suite          → already a tuple[Stmt, ...]
          * bar_closed_stmt → a single Stmt wrapped into a 1-tuple
          * try_stmt       → a TryCatch wrapped into a 1-tuple
        """
        # args always has exactly one relevant child (the grammar guarantees it).
        child = args[0]
        if isinstance(child, tuple):
            return cast(tuple[syntax.Stmt, ...], child)
        return (cast(syntax.Stmt, child),)

    # ------------------------------------------------------------------
    # M3: case_stmt
    # ------------------------------------------------------------------

    def case_stmt(self, meta: Meta, args: _Args) -> syntax.CaseStmt:
        # Grammar: "case" expr "of" (PIPE case_stmt_branch)+
        subject = cast(syntax.Expr, args[0])
        branches = tuple(a for a in args[1:] if isinstance(a, syntax.CaseStmtBranch))
        return syntax.CaseStmt(
            subject=subject, branches=branches,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    def case_stmt_branch(self, meta: Meta, args: _Args) -> syntax.CaseStmtBranch:
        # args: [pattern, tuple[Stmt,...]]
        pat = cast(syntax.Pattern, args[0])
        body = _find_branch_body(args[1:])
        return syntax.CaseStmtBranch(
            pattern=pat, body=body,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # M3: try_stmt / catch_clause
    # ------------------------------------------------------------------

    def try_body(self, meta: Meta, args: _Args) -> tuple[syntax.Stmt, ...]:
        """try_body: suite | closed_stmt (SEMICOLON closed_stmt)*"""
        for a in args:
            if isinstance(a, tuple):
                return cast(tuple[syntax.Stmt, ...], a)
        # Inline form: list of closed stmts
        return tuple(
            cast(syntax.Stmt, a) for a in args
            if a is not None and not isinstance(a, Token) and isinstance(a, (
                syntax.LetDecl, syntax.VarDecl, syntax.SetStmt, syntax.PassStmt,
                syntax.PrintStmt, syntax.ExprStmt, syntax.Raise, syntax.DoUntil,
            ))
        )

    def catch_type_pattern(self, meta: Meta, args: _Args) -> tuple[str | None, str | None]:
        """catch_pattern: TYPE_NAME ("as" VAR_NAME)?"""
        type_tok = next((a for a in args if isinstance(a, Token) and a.type == "TYPE_NAME"), None)
        bind_tok = next((a for a in args if isinstance(a, Token) and a.type == "VAR_NAME"), None)
        exc_type = str(type_tok) if type_tok else None
        binding = str(bind_tok) if bind_tok else None
        return (exc_type, binding)

    def catch_wildcard_pattern(
        self, meta: Meta, args: _Args
    ) -> tuple[str | None, str | None]:
        """catch_pattern: VAR_NAME ("as" VAR_NAME)?  where first VAR_NAME is "_"."""
        var_toks = [a for a in args if isinstance(a, Token) and a.type == "VAR_NAME"]
        wildcard_tok = var_toks[0]
        if str(wildcard_tok) != "_":
            raise AglSyntaxError(
                f"{str(wildcard_tok)!r} is not an exception type name. "
                "Catch patterns take an exception type (capitalized) or '_'.",
                span=_span_from_token(wildcard_tok),
            )
        # First token is the wildcard "_"; second (if present) is the binding name.
        binding: str | None = str(var_toks[1]) if len(var_toks) >= 2 else None
        return (None, binding)

    def catch_body(self, meta: Meta, args: _Args) -> tuple[syntax.Stmt, ...]:
        """catch_body: suite | bar_closed_stmt

        Returns a flat tuple of statements.  The grammar guarantees exactly one child:
          * suite            → already a tuple[Stmt, ...]
          * bar_closed_stmt  → a single Stmt wrapped into a 1-tuple
        """
        child = args[0]
        if isinstance(child, tuple):
            return cast(tuple[syntax.Stmt, ...], child)
        return (cast(syntax.Stmt, child),)

    def catch_clause(self, meta: Meta, args: _Args) -> syntax.CatchClause:
        # args: [(exc_type, binding), tuple[Stmt,...]]
        # The catch_pattern sub-rule returns (exc_type, binding) — a plain tuple.
        # The catch_body sub-rule returns a tuple[Stmt,...].
        # Distinguish by checking element types.
        exc_type: str | None = None
        binding: str | None = None
        body: tuple[syntax.Stmt, ...] = ()
        for a in args:
            if isinstance(a, tuple):
                # Is it (exc_type, binding) or (Stmt, ...)?
                if len(a) == 2 and (a[0] is None or isinstance(a[0], str)) and (
                    a[1] is None or isinstance(a[1], str)
                ):
                    exc_type, binding = cast(tuple[str | None, str | None], a)
                else:
                    body = cast(tuple[syntax.Stmt, ...], a)
        return syntax.CatchClause(
            exc_type=exc_type, binding=binding, body=body,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    def try_stmt(self, meta: Meta, args: _Args) -> syntax.TryCatch:
        # args: [tuple[Stmt,...] (try_body), CatchClause, ...]
        # All grammar children transform to exactly one of: tuple[Stmt,...] or CatchClause.
        handlers = [a for a in args if isinstance(a, syntax.CatchClause)]
        try_body = next(a for a in args if isinstance(a, tuple))
        return syntax.TryCatch(
            body=cast(tuple[syntax.Stmt, ...], try_body),
            handlers=tuple(handlers),
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # M3: Patterns
    # ------------------------------------------------------------------

    def pat_var_or_wild(
        self, meta: Meta, args: _Args
    ) -> syntax.WildcardPattern | syntax.VarPattern:
        """VAR_NAME → WildcardPattern (when value is "_") or VarPattern."""
        tok = args[0]
        assert isinstance(tok, Token)
        if str(tok) == "_":
            return syntax.WildcardPattern(span=_span_from_meta(meta), node_id=self._next_id())
        return syntax.VarPattern(
            name=str(tok), span=_span_from_meta(meta), node_id=self._next_id()
        )

    def pat_constructor(self, meta: Meta, args: _Args) -> syntax.ConstructorPattern:
        # Grammar: TYPE_NAME (DOT TYPE_NAME)? (LPAR pattern_fields? RPAR)?
        type_toks = [a for a in args if isinstance(a, Token) and a.type == "TYPE_NAME"]
        qualifier: str | None = None
        if len(type_toks) == 2:
            qualifier = str(type_toks[0])
            name = str(type_toks[1])
        else:
            name = str(type_toks[0])
        fields: tuple[syntax.PatternField, ...] = ()
        for a in args:
            if isinstance(a, tuple) and all(isinstance(x, syntax.PatternField) for x in a):
                fields = cast(tuple[syntax.PatternField, ...], a)
        return syntax.ConstructorPattern(
            qualifier=qualifier, name=name, fields=fields,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    def _literal_pattern(
        self,
        literal: syntax.IntLit | syntax.DecimalLit | syntax.BoolLit | syntax.StringLit
        | syntax.NullLit,
        meta: Meta,
    ) -> syntax.LiteralPattern:
        return syntax.LiteralPattern(
            literal=literal, span=_span_from_meta(meta), node_id=self._next_id()
        )

    def pat_lit_int(self, meta: Meta, args: _Args) -> syntax.LiteralPattern:
        tok = args[0]
        assert isinstance(tok, Token)
        lit = syntax.IntLit(
            value=int(str(tok)), span=_span_from_meta(meta), node_id=self._next_id()
        )
        return self._literal_pattern(lit, meta)

    def pat_lit_decimal(self, meta: Meta, args: _Args) -> syntax.LiteralPattern:
        tok = args[0]
        assert isinstance(tok, Token)
        import decimal as _decimal
        lit = syntax.DecimalLit(
            value=_decimal.Decimal(str(tok)),
            span=_span_from_meta(meta), node_id=self._next_id(),
        )
        return self._literal_pattern(lit, meta)

    def pat_lit_true(self, meta: Meta, args: _Args) -> syntax.LiteralPattern:
        lit = syntax.BoolLit(
            value=True, span=_span_from_meta(meta), node_id=self._next_id()
        )
        return self._literal_pattern(lit, meta)

    def pat_lit_false(self, meta: Meta, args: _Args) -> syntax.LiteralPattern:
        lit = syntax.BoolLit(
            value=False, span=_span_from_meta(meta), node_id=self._next_id()
        )
        return self._literal_pattern(lit, meta)

    def pat_lit_null(self, meta: Meta, args: _Args) -> syntax.LiteralPattern:
        lit = syntax.NullLit(span=_span_from_meta(meta), node_id=self._next_id())
        return self._literal_pattern(lit, meta)

    def pat_lit_str(self, meta: Meta, args: _Args) -> syntax.LiteralPattern:
        # args: [StringLit | Template] — the template transformer normalises plain strings.
        # A plain string (no interpolation) becomes StringLit; an interpolated string
        # stays as Template and must be rejected as a pattern literal.
        tmpl = _require_literal_string(
            args[0], "Pattern string literals cannot contain interpolation."
        )
        return self._literal_pattern(tmpl, meta)

    def pattern_fields(self, meta: Meta, args: _Args) -> tuple[syntax.PatternField, ...]:
        return tuple(a for a in args if isinstance(a, syntax.PatternField))

    def pat_field_full(self, meta: Meta, args: _Args) -> syntax.PatternField:
        # VAR_NAME COLON pattern
        name_tok = next((a for a in args if isinstance(a, Token) and a.type == "VAR_NAME"), None)
        assert name_tok is not None
        _pat_types = (
            syntax.WildcardPattern, syntax.LiteralPattern,
            syntax.VarPattern, syntax.ConstructorPattern,
        )
        pat = next((a for a in args if isinstance(a, _pat_types)), None)
        assert pat is not None
        assert isinstance(pat, _pat_types)
        return syntax.PatternField(
            name=str(name_tok), pattern=pat,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    def pat_field_shorthand(self, meta: Meta, args: _Args) -> syntax.PatternField:
        # VAR_NAME — shorthand: name: name
        name_tok = args[0]
        assert isinstance(name_tok, Token)
        name = str(name_tok)
        var_pat = syntax.VarPattern(
            name=name, span=_span_from_meta(meta), node_id=self._next_id()
        )
        return syntax.PatternField(
            name=name, pattern=var_pat,
            span=_span_from_meta(meta), node_id=self._next_id(),
        )

    # ------------------------------------------------------------------
    # template
    # ------------------------------------------------------------------

    def template(self, meta: Meta, args: _Args) -> syntax.Template | syntax.StringLit:
        """Build a Template from its segment children.

        Invariant: ``segments`` never contains empty ``TextSegment`` nodes.
        Adjacent interpolations (e.g. ``"${a}${b}"``) and leading/trailing
        interpolations produce empty text fragments in the token stream; an
        empty fragment carries no information, so it is normalized away here.
        Adjacent ``InterpSegment`` nodes are legal.

        Normalisation: a template with no interpolation segments (only text) is
        returned as a ``StringLit``.  This covers simple quoted strings such as
        ``"hello"`` which have no runtime interpolation behaviour.
        """
        # args: TEMPLATE_START, [segments...], TEMPLATE_END
        # Filter out the synthetic tokens; keep only template_segment results,
        # dropping empty TextSegments (they carry no information).
        segments: list[syntax.TemplateSegment] = [
            a
            for a in args
            if isinstance(a, (syntax.TextSegment, syntax.InterpSegment))
            if not (isinstance(a, syntax.TextSegment) and a.text == "")
        ]
        span = _span_from_meta(meta)
        nid = self._next_id()
        # Plain string (no interpolations) → StringLit
        if all(isinstance(s, syntax.TextSegment) for s in segments):
            text = "".join(s.text for s in segments if isinstance(s, syntax.TextSegment))
            return syntax.StringLit(value=text, span=span, node_id=nid)
        return syntax.Template(
            segments=tuple(segments),
            span=span,
            node_id=nid,
        )

    def tmpl_text(self, meta: Meta, args: _Args) -> syntax.TextSegment:
        tok = args[0]
        assert isinstance(tok, Token)
        return syntax.TextSegment(
            text=str(tok),
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def tmpl_interp(self, meta: Meta, args: _Args) -> syntax.InterpSegment:
        # Unwrap the inner interp node.
        (seg,) = args
        assert isinstance(seg, syntax.InterpSegment)
        return seg

    def interp(self, meta: Meta, args: _Args) -> syntax.InterpSegment:
        # Grammar: INTERP_START expr INTERP_END
        # Extract the expression (the only non-token element).
        expr: syntax.Expr = _find_expr(args)
        return syntax.InterpSegment(
            expr=expr,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )


# ---------------------------------------------------------------------------
# Internal helper types (not exported)
# ---------------------------------------------------------------------------


class _FormatOpt:
    key = "format"

    def __init__(self, value: str, span: SourceSpan) -> None:
        self.value = value
        self.span = span


class _StrictJsonOpt:
    key = "strict_json"

    def __init__(self, value: bool, span: SourceSpan) -> None:
        self.value = value
        self.span = span


class _ParseErrorOpt:
    key = "on_parse_error"

    def __init__(self, value: syntax.ParsePolicy, span: SourceSpan) -> None:
        self.value = value
        self.span = span


# ---------------------------------------------------------------------------
# Helper: find first TypeExpr in an args list
# ---------------------------------------------------------------------------


def _find_type_expr(args: _Args) -> TypeExpr:
    """Return the first element that is a TypeExpr instance."""
    for a in args:
        if isinstance(a, (TextT, JsonT, BoolT, IntT, DecimalT, NameT, ListT, DictT)):
            return a
    raise AssertionError(f"_find_type_expr: no TypeExpr found in {args!r}")


def _find_name_token(args: _Args) -> Token:
    """Return the field/key name Token from a ``field_name``-bearing rule.

    ``field_name`` matches either a ``VAR_NAME`` identifier or the reserved word
    ``agent`` (token type ``AGENT``); both arrive here as plain name Tokens.
    The first such Token is the field/key name.
    """
    for a in args:
        if isinstance(a, Token) and a.type in ("VAR_NAME", "AGENT"):
            return a
    raise AssertionError(f"_find_name_token: no name token found in {args!r}")


def _require_literal_string(node: object, message: str) -> syntax.StringLit:
    """Return *node* as a ``StringLit``, rejecting an interpolated ``Template``.

    The ``template`` transformer normalises a plain (non-interpolated) string to
    a ``StringLit`` and keeps an interpolated one as a ``Template``.  Positions
    that require a static string — agent runner hints, dict keys, pattern string
    literals — call this to accept the former and raise ``AglSyntaxError`` with
    *message* on the latter.
    """
    if isinstance(node, syntax.StringLit):
        return node
    assert isinstance(node, syntax.Template), (
        f"_require_literal_string: expected StringLit or Template, got {type(node)}"
    )
    raise AglSyntaxError(message, span=node.span)


def _is_expr_obj(a: object) -> bool:
    """Return True if *a* is an Expr (AST node, not a Token or None)."""
    return a is not None and not isinstance(a, Token)


def _find_expr(args: _Args) -> syntax.Expr:
    """Return the first Expr in *args* (skip Tokens and None placeholders)."""
    for a in args:
        if _is_expr_obj(a):
            return cast(syntax.Expr, a)
    raise AssertionError(f"_find_expr: no Expr found in {args!r}")


def _extract_ann_and_value(
    tail: _Args,
) -> tuple[TypeExpr | None, syntax.Expr]:
    """Extract (type_ann, value) from the tail of a let/var args list.

    Grammar tail is: type_ann? EQ expr
    With maybe_placeholders=True: [None|TypeExpr, Token(EQ), Expr]
    """
    ann: TypeExpr | None = None
    value: syntax.Expr | None = None
    for a in tail:
        if isinstance(a, (TextT, JsonT, BoolT, IntT, DecimalT, NameT, ListT, DictT)):
            ann = a
        elif _is_expr_obj(a) and not isinstance(a, Token):
            value = cast(syntax.Expr, a)
        # None (placeholder) and Token (EQ) are skipped
    assert value is not None, f"_extract_ann_and_value: no Expr found in {tail!r}"
    return ann, value


def syntax_error_from_meta(meta: Meta, message: str) -> AglSyntaxError:
    """Create an AglSyntaxError from a Meta object."""
    return AglSyntaxError(message, span=_span_from_meta(meta))


def _find_branch_body(args: _Args) -> tuple[syntax.Stmt, ...]:
    """Extract the branch_body tuple[Stmt,...] from transformer args.

    ``branch_body`` always returns a ``tuple[Stmt, ...]``; this helper picks it
    out of the args list (which may also contain an expression from the condition
    or an ARROW token, both of which are not tuples).
    """
    stmts = next((a for a in args if isinstance(a, tuple)), None)
    assert stmts is not None, "_find_branch_body: no tuple found in args"
    return cast(tuple[syntax.Stmt, ...], stmts)


# ---------------------------------------------------------------------------
# _NameOrPolicy: intermediate result from opt_val_name
# ---------------------------------------------------------------------------


class _NameOrPolicy:
    """Either a plain name string or an AbortPolicy (from opt_val_name)."""

    def __init__(self, value: str | syntax.AbortPolicy) -> None:
        self.value = value
