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

    The ``node_id`` counter is monotonically increasing and resets per
    ``AstBuilder`` instance.  Each ``parse_program`` call creates a fresh
    builder, so node IDs within a single program are deterministic (assigned
    in tree-walk order — root first, depth-first left-to-right).
    """

    def __init__(self) -> None:
        super().__init__()
        self._counter = count(0)

    def _next_id(self) -> int:
        return next(self._counter)

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
                syntax.InputDecl, syntax.LetDecl, syntax.VarDecl,
                syntax.SetStmt, syntax.PassStmt, syntax.PrintStmt,
                syntax.ExprStmt,
            ))
        )

    # ------------------------------------------------------------------
    # closed_stmt is transparent (?-prefixed via stmt)
    # ------------------------------------------------------------------

    def closed_stmt(self, meta: Meta, args: _Args) -> syntax.Stmt:
        (inner,) = args
        assert isinstance(inner, (
            syntax.RecordDef, syntax.EnumDef, syntax.TypeAlias,
            syntax.InputDecl, syntax.LetDecl, syntax.VarDecl,
            syntax.SetStmt, syntax.PassStmt, syntax.PrintStmt,
            syntax.ExprStmt,
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
        # Grammar: VAR_NAME COLON type_expr
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
    # expr_stmt
    # ------------------------------------------------------------------

    def expr_stmt(self, meta: Meta, args: _Args) -> syntax.ExprStmt:
        expr = _find_expr(args)
        span = _span_from_meta(meta)
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

        In v1 the key type is always ``text``; any other key is rejected with
        the key token's span so the diagnostic pinpoints the offending type.
        """
        # args: [dict, LSQB, key VAR_NAME, COMMA, type_expr, RSQB].  The key is
        # the second VAR_NAME token (the first is the "dict" head).
        var_name_toks = [
            a for a in args if isinstance(a, Token) and a.type == "VAR_NAME"
        ]
        assert len(var_name_toks) >= 2, "dict_type: missing key token"
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
        """access DOT VAR_NAME — record field access (lowercase field name)."""
        obj_expr = cast(syntax.Expr, args[0])
        field_tok = next(
            (a for a in args if isinstance(a, Token) and a.type == "VAR_NAME"), None
        )
        assert field_tok is not None, "field_access: no VAR_NAME token"
        return syntax.FieldAccess(
            obj=obj_expr,
            field=str(field_tok),
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def type_access(self, meta: Meta, args: _Args) -> syntax.Constructor:
        """access DOT TYPE_NAME — qualified enum-variant constructor.

        When ``access`` is an unqualified zero-arg Constructor, produces a
        qualified Constructor: ``Status.Done`` → Constructor("Status","Done",()).
        Any other form is surfaced as a bare Constructor for M2c error reporting.
        """
        obj_expr = args[0]
        type_name_tok = next(
            (a for a in args if isinstance(a, Token) and a.type == "TYPE_NAME"), None
        )
        assert type_name_tok is not None, "type_access: no TYPE_NAME token"
        variant_name = str(type_name_tok)
        span = _span_from_meta(meta)
        nid = self._next_id()
        if isinstance(obj_expr, syntax.Constructor) and obj_expr.qualifier is None:
            return syntax.Constructor(
                qualifier=obj_expr.name,
                name=variant_name,
                args=(),
                span=span,
                node_id=nid,
            )
        return syntax.Constructor(
            qualifier=None,
            name=variant_name,
            args=(),
            span=span,
            node_id=nid,
        )

    def ctor_applied(self, meta: Meta, args: _Args) -> syntax.Constructor:
        """access constructor_payload — apply args to a bare constructor.

        Handles both unqualified ``Type(x: 1)`` and qualified ``Ns.Type(x: 1)``
        by taking the existing (possibly qualified) Constructor from the access
        position and attaching the named arguments from constructor_payload.

        ``args`` is always ``[Constructor, list[NamedArg]]`` since
        ``constructor_payload`` is required (not optional) in this rule.
        """
        ctor = args[0]
        payload = args[1]
        assert isinstance(ctor, syntax.Constructor), (
            f"ctor_applied: expected Constructor base, got {type(ctor)}"
        )
        assert isinstance(payload, list), (
            f"ctor_applied: expected list payload, got {type(payload)}"
        )
        named_args = tuple(cast(_NamedArgList, payload))
        return syntax.Constructor(
            qualifier=ctor.qualifier,
            name=ctor.name,
            args=named_args,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
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
        Both StringLit (plain key) and Template (interpolated key) are accepted;
        only the StringLit form is valid per v1 semantics (DictEntry.key: StringLit).
        """
        non_tokens = [a for a in args if a is not None and not isinstance(a, Token)]
        assert len(non_tokens) >= 2, (
            f"dict_entry_str: expected key + expr, got {args!r}"
        )
        key_node = non_tokens[0]
        val_expr = cast(syntax.Expr, non_tokens[1])
        if isinstance(key_node, syntax.StringLit):
            key_lit = key_node
        else:
            assert isinstance(key_node, syntax.Template), (
                f"dict_entry_str: expected StringLit or Template for key, got {type(key_node)}"
            )
            key_text = "".join(
                seg.text for seg in key_node.segments if isinstance(seg, syntax.TextSegment)
            )
            key_lit = syntax.StringLit(
                value=key_text,
                span=key_node.span,
                node_id=self._next_id(),
            )
        return syntax.DictEntry(
            key=key_lit,
            value=val_expr,
            span=_span_from_meta(meta),
            node_id=self._next_id(),
        )

    def dict_entry_name(self, meta: Meta, args: _Args) -> syntax.DictEntry:
        """dict_entry: VAR_NAME COLON expr — identifier shorthand key.

        The identifier is converted to ``StringLit`` so the AST key is always
        ``StringLit``, matching ``DictEntry.key``.
        """
        name_tok = next(
            (a for a in args if isinstance(a, Token) and a.type == "VAR_NAME"), None
        )
        assert name_tok is not None, "dict_entry_name: no VAR_NAME token"
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
        """named_arg: VAR_NAME COLON expr"""
        name_tok = next(
            (a for a in args if isinstance(a, Token) and a.type == "VAR_NAME"), None
        )
        assert name_tok is not None, "named_arg: no VAR_NAME token"
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
        # Grammar: INTERP_START expr ("as" VAR_NAME)? INTERP_END
        # Lark drops the "as" anonymous terminal from the tree, so args are:
        #   [Token(INTERP_START), <expr-node>, Token?(VAR_NAME renderer), Token(INTERP_END)]
        # The renderer VAR_NAME is the only non-synthetic Token after the expr.
        # Extract the non-token (expr) and any VAR_NAME tokens that follow it.
        expr: syntax.Expr = _find_expr(args)
        render: str | None = None
        # Scan for a VAR_NAME token that appears after the expr in the args list.
        past_expr = False
        for a in args:
            if not isinstance(a, Token):
                past_expr = True
            elif past_expr and a.type == "VAR_NAME":
                render = str(a)
                break
        return syntax.InterpSegment(
            expr=expr,
            render=render,
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


# ---------------------------------------------------------------------------
# _NameOrPolicy: intermediate result from opt_val_name
# ---------------------------------------------------------------------------


class _NameOrPolicy:
    """Either a plain name string or an AbortPolicy (from opt_val_name)."""

    def __init__(self, value: str | syntax.AbortPolicy) -> None:
        self.value = value
