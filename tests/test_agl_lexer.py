"""Tests for the AgL custom lexer.

Tests assert on ``(token_type, value)`` streams from real AgL snippets via the
public ``tokenize`` helper.  No scanner/layout internals are tested.
"""

from __future__ import annotations

import pytest
from lark.lexer import LexerState

from agm.agl.diagnostics import Diagnostic
from agm.agl.lexer import AglLexer, LexError, lex_tab_warnings, tokenize

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def tok(source: str) -> list[tuple[str, str]]:
    """Return ``(type, value)`` pairs for every token in *source*."""
    return [(t.type, str(t)) for t in tokenize(source)]


def lark_tok(source: str) -> list[tuple[str, str]]:
    """Return parser-facing ``(type, value)`` pairs for every token in *source*."""
    lexer = AglLexer(None)
    state = LexerState(source)
    return [(t.type, str(t)) for t in lexer.lex(state, None)]


# ---------------------------------------------------------------------------
# Keywords vs identifiers
# ---------------------------------------------------------------------------


class TestKeywordsAndIdentifiers:
    def test_reserved_keywords_emitted_as_literal_tokens(self) -> None:
        # pass, print, and the former set keyword lex as VAR_NAME.
        result = tok("let var set do until if else case of try catch raise return as")
        types = [t for t, _ in result]
        assert "let" in types
        assert "var" in types
        assert ("NAME", "set") in result
        assert "do" in types
        assert "until" in types
        assert "if" in types
        assert "else" in types
        assert "case" in types
        assert "of" in types
        assert "try" in types
        assert "catch" in types
        assert "raise" in types
        assert "return" in types
        assert "as" in types

    def test_agent_is_reserved_keyword(self) -> None:
        # `agent` is a reserved keyword — its token type is the literal string.
        result = tok("agent")
        assert result == [("agent", "agent")]

    def test_agent_not_var_name(self) -> None:
        # `agent` cannot be used as an identifier/variable name.
        types = [t for t, _ in tok("agent")]
        assert "NAME" not in types

    def test_agent_prefix_identifier(self) -> None:
        # "agentic" starts with "agent" but is a plain identifier.
        result = tok("agentic")
        assert result == [("NAME", "agentic")]

    def test_bool_and_null_keywords(self) -> None:
        result = tok("true false null")
        types = [t for t, _ in result]
        assert "true" in types
        assert "false" in types
        assert "null" in types

    def test_type_name_uppercase(self) -> None:
        result = tok("Review Pass Fail")
        assert result == [
            ("NAME", "Review"),
            ("NAME", "Pass"),
            ("NAME", "Fail"),
        ]

    def test_var_name_lowercase(self) -> None:
        result = tok("artifact review x")
        assert result == [
            ("NAME", "artifact"),
            ("NAME", "review"),
            ("NAME", "x"),
        ]

    def test_var_name_underscore_prefix(self) -> None:
        result = tok("_x _foo")
        types = [t for t, _ in result]
        assert types == ["NAME", "NAME"]

    def test_ask_is_var_name_not_keyword(self) -> None:
        # ask/exec are contextual keywords — lex as VAR_NAME
        result = tok("ask exec")
        assert result == [
            ("NAME", "ask"),
            ("NAME", "exec"),
        ]

    def test_param_is_reserved_keyword(self) -> None:
        # `param` is a reserved keyword — its token type is the literal string.
        result = tok("param")
        assert result == [("param", "param")]

    def test_param_not_var_name(self) -> None:
        types = [t for t, _ in tok("param")]
        assert "NAME" not in types

    def test_program_is_reserved_keyword(self) -> None:
        # `program` is a reserved keyword — its token type is the literal string.
        result = tok("program")
        assert result == [("program", "program")]

    def test_program_not_var_name(self) -> None:
        types = [t for t, _ in tok("program")]
        assert "NAME" not in types

    def test_input_is_now_var_name(self) -> None:
        # `input` is no longer reserved — it lexes as a plain VAR_NAME.
        result = tok("input")
        assert result == [("NAME", "input")]

    def test_keyword_prefix_identifier(self) -> None:
        # "letter" starts with 'l' like "let", but is an identifier
        result = tok("letter")
        assert result == [("NAME", "letter")]

    def test_type_name_mixed_case(self) -> None:
        result = tok("FooBar")
        assert result == [("NAME", "FooBar")]


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


class TestNumbers:
    def test_int_literal(self) -> None:
        result = tok("42")
        assert result == [("INT", "42")]

    def test_decimal_literal(self) -> None:
        result = tok("3.14")
        assert result == [("DECIMAL", "3.14")]

    def test_zero_int(self) -> None:
        result = tok("0")
        assert result == [("INT", "0")]

    def test_zero_decimal(self) -> None:
        result = tok("0.0")
        assert result == [("DECIMAL", "0.0")]

    def test_int_not_decimal(self) -> None:
        # "5" with no dot is INT
        types = [t for t, _ in tok("5")]
        assert types == ["INT"]

    def test_decimal_requires_digits_after_dot(self) -> None:
        # "5." without trailing digit: "5" is INT, "." is DOT
        result = tok("5.")
        assert result[0] == ("INT", "5")
        assert result[1] == ("DOT", ".")


# ---------------------------------------------------------------------------
# Operators — maximal munch
# ---------------------------------------------------------------------------


class TestOperators:
    def test_arrow(self) -> None:
        assert tok("=>") == [("ARROW", "=>")]

    def test_eq(self) -> None:
        assert tok("=") == [("EQ", "=")]

    def test_neq(self) -> None:
        assert tok("!=") == [("NEQ", "!=")]

    def test_le(self) -> None:
        assert tok("<=") == [("LE", "<=")]

    def test_ge(self) -> None:
        assert tok(">=") == [("GE", ">=")]

    def test_lt(self) -> None:
        assert tok("<") == [("LT", "<")]

    def test_gt(self) -> None:
        assert tok(">") == [("GT", ">")]

    def test_plus(self) -> None:
        assert tok("+") == [("PLUS", "+")]

    def test_minus(self) -> None:
        assert tok("-") == [("MINUS", "-")]

    def test_star(self) -> None:
        assert tok("*") == [("STAR", "*")]

    def test_slash(self) -> None:
        assert tok("/") == [("SLASH", "/")]

    def test_lpar(self) -> None:
        assert tok("(") == [("LPAR", "(")]

    def test_rpar(self) -> None:
        assert tok(")") == [("RPAR", ")")]

    def test_lsqb(self) -> None:
        assert tok("[") == [("LSQB", "[")]

    def test_rsqb(self) -> None:
        assert tok("]") == [("RSQB", "]")]

    def test_lbrace(self) -> None:
        assert tok("{") == [("LBRACE", "{")]

    def test_rbrace(self) -> None:
        assert tok("}") == [("RBRACE", "}")]

    def test_colon(self) -> None:
        assert tok(":") == [("COLON", ":")]

    def test_assignment(self) -> None:
        assert tok(":=") == [("ASSIGN", ":=")]

    def test_dcolon(self) -> None:
        # ``::`` is the type-argument introducer for typed calls
        # (``ask-request::[Review](...)``); maximal munch wins over two COLONs.
        assert tok("::") == [("DCOLON", "::")]

    def test_dcolon_maximal_munch_before_colon(self) -> None:
        # A single ``:`` still lexes as COLON even when adjacent to ``::``;
        # the scanner consumes the ``::`` greedily, leaving a lone ``:``.
        assert tok(":: :") == [("DCOLON", "::"), ("COLON", ":")]
        assert tok(": ::") == [("COLON", ":"), ("DCOLON", "::")]

    def test_dcolon_not_part_of_identifier(self) -> None:
        # ``::`` breaks an identifier scan (``:`` is a stop character), so a
        # ``name::`` tail does not glue into the identifier.
        assert tok("ask-request::[Review]") == [
            ("NAME", "ask-request"), ("DCOLON", "::"), ("LSQB", "["),
            ("NAME", "Review"), ("RSQB", "]"),
        ]

    def test_comma(self) -> None:
        assert tok(",") == [("COMMA", ",")]

    def test_dot(self) -> None:
        assert tok(".") == [("DOT", ".")]

    def test_pipe(self) -> None:
        assert tok("|") == [("PIPE", "|")]

    def test_semicolon(self) -> None:
        assert tok(";") == [("SEMICOLON", ";")]

    def test_eq_eq_is_error_token(self) -> None:
        # "==" emits the dedicated EQ_EQ error token (not two EQ tokens)
        result = tok("==")
        assert result == [("EQ_EQ", "==")]

    def test_operator_name_tokens(self) -> None:
        assert tok("==> >> |> <| >=> %$ %? ~ ⊕") == [
            ("OP_NAME", "==>"),
            ("OP_NAME", ">>"),
            ("OP_NAME", "|>"),
            ("OP_NAME", "<|"),
            ("OP_NAME", ">=>"),
            ("OP_NAME", "%$"),
            ("OP_NAME", "%?"),
            ("OP_NAME", "~"),
            ("OP_NAME", "⊕"),
        ]

    def test_reserved_operator_tokens_stay_syntax(self) -> None:
        assert tok("= == != <= < >= > -> => := :: + - * / | @ . :") == [
            ("EQ", "="),
            ("EQ_EQ", "=="),
            ("NEQ", "!="),
            ("LE", "<="),
            ("LT", "<"),
            ("GE", ">="),
            ("GT", ">"),
            ("THIN_ARROW", "->"),
            ("ARROW", "=>"),
            ("ASSIGN", ":="),
            ("DCOLON", "::"),
            ("PLUS", "+"),
            ("MINUS", "-"),
            ("STAR", "*"),
            ("SLASH", "/"),
            ("PIPE", "|"),
            ("AT", "@"),
            ("DOT", "."),
            ("COLON", ":"),
        ]

    def test_maximal_munch_arrow_vs_eq(self) -> None:
        # "=>" must be ARROW, not EQ + GT
        result = tok("=>")
        assert len(result) == 1
        assert result[0][0] == "ARROW"

    def test_maximal_munch_le_vs_lt(self) -> None:
        result = tok("<=")
        assert result == [("LE", "<=")]

    def test_maximal_munch_ge_vs_gt(self) -> None:
        result = tok(">=")
        assert result == [("GE", ">=")]

    def test_maximal_munch_neq_vs_bang(self) -> None:
        result = tok("!=")
        assert result == [("NEQ", "!=")]


class TestIndexBracketRemap:
    def test_adjacent_lsqb_after_expression_ending_token_is_index_lsqb(self) -> None:
        assert lark_tok('xs[0] d["a"] make()[0] [1][0] {"a": 1}["a"]') == [
            ("NAME", "xs"),
            ("INDEX_LSQB", "["),
            ("INT", "0"),
            ("RSQB", "]"),
            ("NAME", "d"),
            ("INDEX_LSQB", "["),
            ("TEMPLATE_START", '"'),
            ("STRING_FRAGMENT", "a"),
            ("TEMPLATE_END", '"'),
            ("RSQB", "]"),
            ("NAME", "make"),
            ("LPAR", "("),
            ("RPAR", ")"),
            ("INDEX_LSQB", "["),
            ("INT", "0"),
            ("RSQB", "]"),
            ("LSQB", "["),
            ("INT", "1"),
            ("RSQB", "]"),
            ("INDEX_LSQB", "["),
            ("INT", "0"),
            ("RSQB", "]"),
            ("LBRACE", "{"),
            ("TEMPLATE_START", '"'),
            ("STRING_FRAGMENT", "a"),
            ("TEMPLATE_END", '"'),
            ("COLON", ":"),
            ("INT", "1"),
            ("RBRACE", "}"),
            ("INDEX_LSQB", "["),
            ("TEMPLATE_START", '"'),
            ("STRING_FRAGMENT", "a"),
            ("TEMPLATE_END", '"'),
            ("RSQB", "]"),
        ]

    def test_adjacent_lbrace_after_expression_ending_token_is_call_lbrace(self) -> None:
        assert lark_tok('Point{x: 1} {"x": 1}') == [
            ("NAME", "Point"),
            ("CALL_LBRACE", "{"),
            ("NAME", "x"),
            ("COLON", ":"),
            ("INT", "1"),
            ("RBRACE", "}"),
            ("LBRACE", "{"),
            ("TEMPLATE_START", '"'),
            ("STRING_FRAGMENT", "x"),
            ("TEMPLATE_END", '"'),
            ("COLON", ":"),
            ("INT", "1"),
            ("RBRACE", "}"),
        ]

    def test_spaced_lsqb_stays_list_literal_lsqb(self) -> None:
        assert lark_tok("f [2]") == [
            ("NAME", "f"),
            ("LSQB", "["),
            ("INT", "2"),
            ("RSQB", "]"),
        ]

    def test_newline_and_comment_separated_lsqb_stays_lsqb(self) -> None:
        assert lark_tok("xs\n[0]") == [
            ("NAME", "xs"),
            ("_NEWLINE", "0"),
            ("LSQB", "["),
            ("INT", "0"),
            ("RSQB", "]"),
        ]
        assert lark_tok("xs # comment\n[0]") == [
            ("NAME", "xs"),
            ("_NEWLINE", "0"),
            ("LSQB", "["),
            ("INT", "0"),
            ("RSQB", "]"),
        ]

    def test_do_loop_bound_merge_still_wins_over_index_lsqb(self) -> None:
        assert lark_tok("do[3] tick until done") == [
            ("DO", "do"),
            ("DO_LSQB", "["),
            ("INT", "3"),
            ("RSQB", "]"),
            ("NAME", "tick"),
            ("UNTIL", "until"),
            ("DONE", "done"),
        ]

    def test_qual_typed_call_lsqb_is_index_lsqb(self) -> None:
        # lib::Box::[int](value: 1) — the [ after :: stays LSQB (not adjacent to NAME).
        # lib::Box[0] — the [ adjacent to NAME becomes INDEX_LSQB (index access).
        typed_call = lark_tok("lib::Box::[int](value: 1)")
        assert ("LSQB", "[") in typed_call
        assert ("INDEX_LSQB", "[") not in typed_call

        index = lark_tok("lib::xs[0]")
        assert ("INDEX_LSQB", "[") in index


# ---------------------------------------------------------------------------
# Simple templates (single-line strings)
# ---------------------------------------------------------------------------


class TestSimpleTemplates:
    def test_empty_string(self) -> None:
        result = tok('""')
        assert result == [
            ("TEMPLATE_START", '"'),
            ("STRING_FRAGMENT", ""),
            ("TEMPLATE_END", '"'),
        ]

    def test_simple_string_literal(self) -> None:
        result = tok('"hello"')
        assert result == [
            ("TEMPLATE_START", '"'),
            ("STRING_FRAGMENT", "hello"),
            ("TEMPLATE_END", '"'),
        ]

    def test_single_interpolation(self) -> None:
        result = tok('"hello ${name}"')
        assert result == [
            ("TEMPLATE_START", '"'),
            ("STRING_FRAGMENT", "hello "),
            ("INTERP_START", "${"),
            ("NAME", "name"),
            ("INTERP_END", "}"),
            ("STRING_FRAGMENT", ""),
            ("TEMPLATE_END", '"'),
        ]

    def test_multi_interpolation(self) -> None:
        result = tok('"${a} and ${b}"')
        assert result == [
            ("TEMPLATE_START", '"'),
            ("STRING_FRAGMENT", ""),
            ("INTERP_START", "${"),
            ("NAME", "a"),
            ("INTERP_END", "}"),
            ("STRING_FRAGMENT", " and "),
            ("INTERP_START", "${"),
            ("NAME", "b"),
            ("INTERP_END", "}"),
            ("STRING_FRAGMENT", ""),
            ("TEMPLATE_END", '"'),
        ]

    def test_dollar_not_followed_by_brace_is_literal(self) -> None:
        result = tok('"$x"')
        frags = [(t, v) for t, v in result if t == "STRING_FRAGMENT"]
        assert len(frags) == 1
        assert frags[0] == ("STRING_FRAGMENT", "$x")

    def test_escaped_dollar(self) -> None:
        # \$ produces a literal dollar in the fragment
        result = tok(r'"\$"')
        frags = [(t, v) for t, v in result if t == "STRING_FRAGMENT"]
        assert len(frags) == 1
        assert frags[0] == ("STRING_FRAGMENT", "$")

    def test_escape_sequences(self) -> None:
        # JSON set: \" \\ \/ \b \f \n \r \t
        result = tok(r'"\"\\\/ \b\f\n\r\t"')
        frags = [(t, v) for t, v in result if t == "STRING_FRAGMENT"]
        assert len(frags) == 1
        assert frags[0] == ("STRING_FRAGMENT", '"\\/ \b\f\n\r\t')

    def test_unicode_escape(self) -> None:
        # A is the unicode escape for 'A'
        result = tok('"\\u0041"')
        frags = [(t, v) for t, v in result if t == "STRING_FRAGMENT"]
        assert len(frags) == 1
        assert frags[0] == ("STRING_FRAGMENT", "A")

    def test_unknown_escape_raises_lex_error(self) -> None:
        with pytest.raises(LexError) as exc_info:
            tok(r'"\q"')
        assert exc_info.value.span is not None

    def test_unknown_escape_message_renders_escape_plainly(self) -> None:
        # The bad escape is rendered as ``\q`` (a literal backslash + char),
        # not as ``\'q'`` (the Python repr of the offending character).
        with pytest.raises(LexError) as exc_info:
            tok(r'"\q"')
        msg = str(exc_info.value)
        assert r"\q" in msg
        assert "'q'" not in msg

    def test_nested_braces_inside_interpolation(self) -> None:
        # nested { } inside ${ } should not prematurely close the interpolation
        result = tok('"${foo}"')
        types = [t for t, _ in result]
        assert "INTERP_START" in types
        assert "INTERP_END" in types

    def test_expression_tokens_inside_interp(self) -> None:
        result = tok('"${x + 1}"')
        inner_types = []
        in_interp = False
        for t, v in result:
            if t == "INTERP_START":
                in_interp = True
            elif t == "INTERP_END":
                in_interp = False
            elif in_interp:
                inner_types.append(t)
        assert "NAME" in inner_types
        assert "PLUS" in inner_types
        assert "INT" in inner_types


# ---------------------------------------------------------------------------
# Nested braces inside interpolation
# ---------------------------------------------------------------------------


class TestNestedBracesInInterp:
    def test_lbrace_rbrace_inside_interp(self) -> None:
        # "${foo}" — the VAR_NAME is seen, not a dict literal brace confusion
        result = tok('"${foo}"')
        types = [t for t, _ in result]
        assert types.index("INTERP_START") < types.index("NAME")
        assert types.index("NAME") < types.index("INTERP_END")

    def test_dict_literal_inside_interp(self) -> None:
        # "${{}}" — empty dict literal inside interpolation
        result = tok('"${{}}"')
        types = [t for t, _ in result]
        # Should have INTERP_START, LBRACE, RBRACE, INTERP_END
        assert "INTERP_START" in types
        assert "LBRACE" in types
        assert "RBRACE" in types
        assert "INTERP_END" in types
        # The INTERP_END must come after the inner RBRACE
        last_interp_end = max(i for i, t in enumerate(types) if t == "INTERP_END")
        last_rbrace = max(i for i, t in enumerate(types) if t == "RBRACE")
        assert last_interp_end > last_rbrace


# ---------------------------------------------------------------------------
# Triple-quoted strings (dedent rule)
# ---------------------------------------------------------------------------


class TestTripleQuotedStrings:
    def test_simple_triple_quoted(self) -> None:
        source = '"""hello"""'
        result = tok(source)
        frags = [(t, v) for t, v in result if t == "STRING_FRAGMENT"]
        assert len(frags) == 1
        assert frags[0][1] == "hello"

    def test_triple_quoted_leading_newline_stripped(self) -> None:
        source = '"""\nhello\n"""'
        result = tok(source)
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        content = "".join(frags)
        # One leading newline and one trailing newline are stripped
        assert content == "hello"

    def test_triple_quoted_dedent(self) -> None:
        # Minimum common indentation of non-blank lines is stripped
        source = '"""\n    hello\n    world\n    """'
        result = tok(source)
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        content = "".join(frags)
        assert content == "hello\nworld"

    def test_triple_quoted_mixed_indentation(self) -> None:
        # Minimum indentation is 4, extra indentation on second line is preserved
        source = '"""\n    hello\n      world\n    """'
        result = tok(source)
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        content = "".join(frags)
        assert content == "hello\n  world"

    def test_triple_quoted_blank_lines_not_dedented(self) -> None:
        # Blank lines don't contribute to minimum indentation calculation
        source = '"""\n    hello\n\n    world\n    """'
        result = tok(source)
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        content = "".join(frags)
        # Blank line preserved (but leading/trailing newlines stripped)
        assert content == "hello\n\nworld"

    def test_triple_quoted_with_interpolation(self) -> None:
        source = '"""\nhello ${name}\n"""'
        result = tok(source)
        types = [t for t, _ in result]
        assert "INTERP_START" in types
        assert "INTERP_END" in types

    def test_triple_quoted_interp_hole_not_dedented(self) -> None:
        # Interpolation holes occupy their line; values inside are never dedented
        # The literal skeleton around interp holes gets dedented, not the interp tokens
        source = '"""\n    hello\n    ${x}\n    world\n    """'
        result = tok(source)
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        # Content before interp should be "hello\n"
        # Content after interp should be "\nworld"
        assert frags[0] == "hello\n"
        assert frags[-1] == "\nworld"

    def test_triple_quoted_no_trailing_newline_preserved(self) -> None:
        # No trailing newline before """ is OK
        source = '"""hello"""'
        result = tok(source)
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        content = "".join(frags)
        assert content == "hello"


# ---------------------------------------------------------------------------
# Single-quoted strings
# ---------------------------------------------------------------------------


class TestSingleQuotedStrings:
    def test_empty_single_quoted_string(self) -> None:
        result = tok("''")
        assert result == [
            ("TEMPLATE_START", "'"),
            ("STRING_FRAGMENT", ""),
            ("TEMPLATE_END", "'"),
        ]

    def test_simple_single_quoted_string(self) -> None:
        result = tok("'hello'")
        assert result == [
            ("TEMPLATE_START", "'"),
            ("STRING_FRAGMENT", "hello"),
            ("TEMPLATE_END", "'"),
        ]

    def test_double_quote_inside_single_quoted_string(self) -> None:
        result = tok('\'"\'')
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        assert frags == ['"']

    def test_escaped_single_quote_in_single_quoted_string(self) -> None:
        result = tok(r"'\''")
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        assert frags == ["'"]

    def test_single_quoted_with_interpolation(self) -> None:
        result = tok("'hello ${name}'")
        assert result == [
            ("TEMPLATE_START", "'"),
            ("STRING_FRAGMENT", "hello "),
            ("INTERP_START", "${"),
            ("NAME", "name"),
            ("INTERP_END", "}"),
            ("STRING_FRAGMENT", ""),
            ("TEMPLATE_END", "'"),
        ]

    def test_triple_single_quoted_string(self) -> None:
        source = "'''hello'''"
        result = tok(source)
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        assert "".join(frags) == "hello"

    def test_triple_single_quoted_with_dedent(self) -> None:
        source = "'''\n    hello\n    '''"
        result = tok(source)
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        assert "".join(frags) == "hello"

    def test_escaped_double_quote_in_single_quoted_string(self) -> None:
        result = tok(r'"\""')
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        assert frags == ['"']

    def test_unterminated_single_quoted_string(self) -> None:
        with pytest.raises(LexError) as exc_info:
            tok("'hello")
        assert exc_info.value.span is not None

    def test_newline_inside_single_quoted_string(self) -> None:
        with pytest.raises(LexError) as exc_info:
            tok("'hello\nworld'")
        assert exc_info.value.span is not None

    def test_unterminated_triple_single_quoted_string(self) -> None:
        with pytest.raises(LexError) as exc_info:
            tok("'''hello")
        assert exc_info.value.span is not None


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------


class TestComments:
    def test_line_end_comment_ignored(self) -> None:
        result = tok("let x = 1  # this is a comment")
        types = [t for t, _ in result]
        # No token for the comment
        assert "COMMENT" not in types
        assert "let" in types
        assert "INT" in types

    def test_full_line_comment_ignored(self) -> None:
        # A full-line comment should not produce any tokens
        result = tok("# this is a comment")
        # No tokens (or just layout) expected for a file with only a comment
        non_layout = [(t, v) for t, v in result if not t.startswith("_")]
        assert non_layout == []

    def test_comment_before_dedent(self) -> None:
        # Comment on a line before a dedent: the _NEWLINE after the comment
        # should reflect the NEXT real line's indentation
        source = "if x =>\n  pass\n  # comment\nend"
        # We just want no crash and proper token stream
        result = tok(source)
        types = [t for t, _ in result]
        # Should have INDENT and DEDENT around the body
        assert "_INDENT" in types
        assert "_DEDENT" in types

    def test_comment_on_dedent_line_reflects_next_indentation(self) -> None:
        # Two levels: after the comment, the next real line is at column 0
        source = "if x =>\n  pass\n  # comment\nother"
        result = tok(source)
        types = [t for t, _ in result]
        assert "_INDENT" in types
        assert "_DEDENT" in types


# ---------------------------------------------------------------------------
# Layout: INDENT / DEDENT
# ---------------------------------------------------------------------------


class TestLayout:
    def test_simple_indent(self) -> None:
        source = "if x =>\n  pass"
        result = tok(source)
        types = [t for t, _ in result]
        assert "_INDENT" in types

    def test_indent_and_dedent(self) -> None:
        source = "if x =>\n  pass\nother"
        result = tok(source)
        types = [t for t, _ in result]
        assert "_INDENT" in types
        assert "_DEDENT" in types

    def test_dedent_order(self) -> None:
        # outer line at col 0, inner at col 2
        source = "a\n  b\nc"
        result = tok(source)
        types = [t for t, _ in result]
        indent_idx = types.index("_INDENT")
        dedent_idx = types.index("_DEDENT")
        assert indent_idx < dedent_idx

    def test_multiple_dedents(self) -> None:
        source = "a\n  b\n    c\nd"
        result = tok(source)
        types = [t for t, _ in result]
        assert types.count("_INDENT") == 2
        assert types.count("_DEDENT") == 2

    def test_bracket_suppresses_newlines(self) -> None:
        # Inside (), _NEWLINE tokens are suppressed
        source = "(\na\nb\n)"
        result = tok(source)
        types = [t for t, _ in result]
        assert "_NEWLINE" not in types
        assert "_INDENT" not in types
        assert "_DEDENT" not in types

    def test_square_bracket_suppresses_newlines(self) -> None:
        source = "[\na\nb\n]"
        result = tok(source)
        types = [t for t, _ in result]
        assert "_NEWLINE" not in types

    def test_curly_brace_suppresses_newlines(self) -> None:
        source = "{\na\nb\n}"
        result = tok(source)
        types = [t for t, _ in result]
        assert "_NEWLINE" not in types

    def test_misaligned_dedent_raises_lex_error(self) -> None:
        # Indent to col 4, then dedent to col 2 which is not on the stack
        source = "a\n    b\n  c"
        with pytest.raises(LexError) as exc_info:
            tok(source)
        assert exc_info.value.span is not None

    def test_eof_unwinds_dedents(self) -> None:
        source = "a\n  b"
        result = tok(source)
        types = [t for t, _ in result]
        # EOF should emit DEDENT to close the indent
        assert "_DEDENT" in types


# ---------------------------------------------------------------------------
# Inline semicolons
# ---------------------------------------------------------------------------


class TestInlineSemicolons:
    def test_semicolon_as_separator(self) -> None:
        result = tok("let x = 1; let y = 2")
        types = [t for t, _ in result]
        assert "SEMICOLON" in types
        assert types.count("let") == 2

    def test_semicolons_and_indent_mixed(self) -> None:
        source = "let x = 1\n  let y = 2; let z = 3"
        result = tok(source)
        types = [t for t, _ in result]
        assert "SEMICOLON" in types
        assert "_INDENT" in types


# ---------------------------------------------------------------------------
# Pipe (|) continuation rule
# ---------------------------------------------------------------------------


class TestPipeContinuation:
    def test_pipe_at_same_column_suppresses_newline(self) -> None:
        # if/| branches at same indentation — no interior _NEWLINE
        source = "if x =>\n  pass\n| else =>\n  pass"
        result = tok(source)
        types = [t for t, _ in result]
        # The | branches should not be separated by _NEWLINE
        # Between the DEDENT of first branch and the | there should be no _NEWLINE
        pipe_indices = [i for i, t in enumerate(types) if t == "PIPE"]
        assert len(pipe_indices) == 1
        # No _NEWLINE should appear just before a PIPE (after potential DEDENTs)
        for pipe_idx in pipe_indices:
            # Check that no _NEWLINE precedes this PIPE
            # (there may be _DEDENT tokens between)
            for j in range(pipe_idx - 1, -1, -1):
                if types[j] == "_NEWLINE":
                    pytest.fail(f"_NEWLINE found at {j} before PIPE at {pipe_idx}")
                    break
                if types[j] not in ("_DEDENT",):
                    break

    def test_pipe_at_same_column_only_dedents_needed(self) -> None:
        # Nested case: outer | at column 0, inner body was indented
        source = "case x of\n  | A =>\n    pass\n  | B =>\n    pass"
        result = tok(source)
        types = [t for t, _ in result]
        pipe_indices = [i for i, t in enumerate(types) if t == "PIPE"]
        assert len(pipe_indices) == 2
        # All pipes after the first should not have _NEWLINE before them
        for pipe_idx in pipe_indices[1:]:
            for j in range(pipe_idx - 1, -1, -1):
                if types[j] == "_NEWLINE":
                    pytest.fail(f"Interior _NEWLINE before PIPE at {pipe_idx}")
                    break
                if types[j] not in ("_DEDENT",):
                    break

    def test_outer_pipe_pops_inner_indentation(self) -> None:
        # Deeper pipe at outer level pops inner indentation
        # if at col 0, branch body indented to col 2
        # outer | at col 0 => pops col 2 with DEDENT
        source = "if x =>\n  pass\n| else =>\n  pass"
        result = tok(source)
        types = [t for t, _ in result]
        # The outer | should have been preceded by a DEDENT (closing the inner body)
        pipe_idx = next(i for i, t in enumerate(types) if t == "PIPE")
        assert "_DEDENT" in types[: pipe_idx + 1]

    def test_indented_pipe_no_extra_indent(self) -> None:
        # Pipe at a deeper level than the case keyword.
        # The |‑continuation rule makes the branch list flat:
        # no _INDENT/_DEDENT wrapping the branch list, no extra INDENTs for | lines.
        source = "case x of\n  | A => pass\n  | B => pass"
        result = tok(source)
        types = [t for t, _ in result]
        # No _INDENT at all: the | branches are flat (|‑continuation rule)
        assert types.count("_INDENT") == 0

    def test_pipe_does_not_push_indent(self) -> None:
        # A | line never pushes an INDENT
        source = "if a =>\n  pass\n| b =>\n  other\n| else =>\n  finish"
        result = tok(source)
        types = [t for t, _ in result]
        # Only the branch bodies cause indents
        assert types.count("_INDENT") == 3  # 3 branch bodies
        # The | tokens themselves don't push indents
        pipe_indices = [i for i, t in enumerate(types) if t == "PIPE"]
        # A _INDENT must never immediately follow a | (a | line never pushes).
        for pipe_idx in pipe_indices:
            assert types[pipe_idx + 1] != "_INDENT"

    def test_pipe_continuation_inline_branches(self) -> None:
        # All branches inline: no _NEWLINE anywhere
        source = "if a => pass | else => pass"
        result = tok(source)
        types = [t for t, _ in result]
        assert "_NEWLINE" not in types
        assert "_INDENT" not in types

    def test_catch_suppresses_newline(self) -> None:
        # ``catch`` at the start of a line suppresses the preceding _NEWLINE
        # so the ``try`` body and catch
        # clause are lexically joined without an interior _NEWLINE.
        source = "try\n  pass\ncatch _ =>\n  pass"
        result = tok(source)
        types = [t for t, _ in result]
        catch_idx = next(i for i, t in enumerate(types) if t == "catch")
        # No _NEWLINE should appear in the tokens leading up to ``catch``
        # (only _DEDENT tokens may precede it after the try body).
        for j in range(catch_idx - 1, -1, -1):
            assert types[j] != "_NEWLINE", (
                f"_NEWLINE found at position {j} before 'catch' at {catch_idx}"
            )
            if types[j] not in ("_DEDENT",):
                break

    def test_else_suppresses_newline(self) -> None:
        # ``else`` may omit its leading pipe, so at the start of a line it
        # continues the enclosing if expression like ``|`` does.
        source = "if x =>\n  pass\nelse =>\n  pass"
        result = tok(source)
        types = [t for t, _ in result]
        else_idx = next(i for i, t in enumerate(types) if t == "else")
        for j in range(else_idx - 1, -1, -1):
            assert types[j] != "_NEWLINE", (
                f"_NEWLINE found at position {j} before 'else' at {else_idx}"
            )
            if types[j] not in ("_DEDENT",):
                break

    def test_until_suppresses_newline(self) -> None:
        # ``until`` at the start of a line suppresses the preceding _NEWLINE
        # so the do body and condition
        # are lexically joined without an interior _NEWLINE.
        source = "do[2]\n  pass\nuntil true"
        result = tok(source)
        types = [t for t, _ in result]
        until_idx = next(i for i, t in enumerate(types) if t == "until")
        # No _NEWLINE should appear in the tokens leading up to ``until``
        for j in range(until_idx - 1, -1, -1):
            assert types[j] != "_NEWLINE", (
                f"_NEWLINE found at position {j} before 'until' at {until_idx}"
            )
            if types[j] not in ("_DEDENT",):
                break


# ---------------------------------------------------------------------------
# _NEWLINE token value (indentation width)
# ---------------------------------------------------------------------------


class TestNewlineTokenValue:
    def test_newline_carries_zero_indentation_at_same_level(self) -> None:
        # When the next line is at the same level (0), _NEWLINE passes through with value "0"
        source = "a\nb"
        result = tok(source)
        newlines = [(t, v) for t, v in result if t == "_NEWLINE"]
        assert len(newlines) >= 1
        assert newlines[0][1] == "0"

    def test_newline_becomes_indent_when_deeper(self) -> None:
        # When the next line is deeper, _NEWLINE becomes _INDENT (not passed through)
        source = "a\n  b"
        result = tok(source)
        types = [t for t, _ in result]
        assert "_INDENT" in types
        # No _NEWLINE for this transition (it's consumed to produce _INDENT)
        assert "_NEWLINE" not in types

    def test_tab_expansion_causes_indent(self) -> None:
        # Tab expands to tab_len=4 boundary; a single tab at col 0 gives width 4
        # which is deeper than 0, so _NEWLINE becomes _INDENT
        source = "a\n\tb"
        result = tok(source)
        types = [t for t, _ in result]
        assert "_INDENT" in types


# ---------------------------------------------------------------------------
# Position info
# ---------------------------------------------------------------------------


class TestTokenPositions:
    def test_token_has_line_and_column(self) -> None:
        tokens = list(tokenize("let x = 1"))
        let_tok = next(t for t in tokens if t.type == "let")
        assert let_tok.line == 1
        assert let_tok.column == 1

    def test_token_column_advances(self) -> None:
        tokens = list(tokenize("let x"))
        x_tok = next(t for t in tokens if t.type == "NAME" and str(t) == "x")
        assert x_tok.line == 1
        assert x_tok.column == 5  # "let " = 4 chars, x at col 5

    def test_token_line_advances(self) -> None:
        tokens = list(tokenize("a\nb"))
        b_tok = next(t for t in tokens if t.type == "NAME" and str(t) == "b")
        assert b_tok.line == 2

    def test_token_end_position(self) -> None:
        tokens = list(tokenize("let"))
        let_tok = tokens[0]
        assert let_tok.end_pos is not None
        assert let_tok.start_pos is not None
        assert let_tok.end_pos == let_tok.start_pos + 3


# ---------------------------------------------------------------------------
# LexError span
# ---------------------------------------------------------------------------


class TestLexErrorSpan:
    def test_unknown_escape_span_line(self) -> None:
        with pytest.raises(LexError) as exc_info:
            tok(r'"\q"')
        err = exc_info.value
        assert err.span is not None
        assert err.span.start_line == 1

    def test_misaligned_dedent_span(self) -> None:
        with pytest.raises(LexError) as exc_info:
            tok("a\n    b\n  c")
        err = exc_info.value
        assert err.span is not None
        # The diagnostic is positioned at the first token on the offending line
        # (the lookahead ``sig`` token) so that the reported line matches the
        # line the user actually misindented — line 3 in this case ("  c").
        assert err.span.start_line == 3

    def test_lex_error_message_not_empty(self) -> None:
        with pytest.raises(LexError) as exc_info:
            tok(r'"\q"')
        assert str(exc_info.value) != ""

    def test_unterminated_string_at_eof(self) -> None:
        # Single-quoted string without closing quote
        with pytest.raises(LexError) as exc_info:
            tok('"hello')
        assert exc_info.value.span is not None

    def test_newline_inside_single_quoted_string(self) -> None:
        # Newline inside a single-quoted string is a lex error
        with pytest.raises(LexError) as exc_info:
            tok('"hello\nworld"')
        assert exc_info.value.span is not None

    def test_unterminated_interpolation(self) -> None:
        # ${ without any closing } — EOF inside interp
        with pytest.raises(LexError) as exc_info:
            tok('"${')
        assert exc_info.value.span is not None

    def test_unterminated_triple_quoted_string(self) -> None:
        # triple-quoted without closing triple-quote
        with pytest.raises(LexError) as exc_info:
            tok('"""hello')
        assert exc_info.value.span is not None

    def test_backslash_at_end_of_string(self) -> None:
        # \\ at EOF inside a string
        with pytest.raises(LexError) as exc_info:
            tok('"\\')  # starts a template, sees \, then EOF
        assert exc_info.value.span is not None

    def test_incomplete_unicode_escape(self) -> None:
        # \u followed by only 2 hex digits then EOF (no closing quote)
        with pytest.raises(LexError) as exc_info:
            tok('"\\u00')
        assert exc_info.value.span is not None

    def test_invalid_unicode_escape_hex_digit(self) -> None:
        # \u followed by a non-hex character
        with pytest.raises(LexError) as exc_info:
            tok('"\\uXXXX"')
        assert exc_info.value.span is not None

    def test_unknown_character_raises_lex_error(self) -> None:
        # Format/control characters are not valid AgL code tokens.
        with pytest.raises(LexError) as exc_info:
            tok("\u200b")
        assert exc_info.value.span is not None

    def test_triple_quoted_with_escape(self) -> None:
        # Escape inside a triple-quoted string
        result = tok('"""\\$hello"""')
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        content = "".join(frags)
        assert content == "$hello"

    def test_eof_after_newline_no_content(self) -> None:
        # Trailing whitespace-only after newline (EOF during indentation
        # measurement).  Only the statement and a trailing _NEWLINE (level 0)
        # are produced -- no _INDENT/_DEDENT noise from the blank tail.
        result = tok("a\n   ")
        assert result == [("NAME", "a"), ("_NEWLINE", "0")]

    def test_peek_significant_eof_after_newline(self) -> None:
        # A trailing newline at EOF yields the statement plus a single trailing
        # _NEWLINE at level 0 (which the grammar's optional trailing separator
        # consumes); no _INDENT/_DEDENT is emitted.
        result = tok("a\n")
        assert result == [("NAME", "a"), ("_NEWLINE", "0")]

    def test_comment_at_eof_no_trailing_newline(self) -> None:
        # After a statement and newline, a comment that extends to EOF (no trailing newline)
        # Tests the _measure_indentation path where comment hits EOF without a newline.
        result = tok("a\n# comment at eof")
        types = [t for t, _ in result]
        assert "NAME" in types


# ---------------------------------------------------------------------------
# Real AgL snippet integration tests
# ---------------------------------------------------------------------------


class TestAglLexerClass:
    """Tests for the AglLexer Lark interface."""

    def test_agl_lexer_lex_method(self) -> None:
        # AglLexer.lex is the Lark parser interface.  Keyword types are
        # remapped to their uppercase grammar terminal names (e.g. "let" →
        # "LET") so the LALR parse table can resolve them.
        lexer = AglLexer(None)
        state = LexerState("let x = 1")
        result = list(lexer.lex(state, None))
        types = [t.type for t in result]
        assert "LET" in types
        assert "NAME" in types
        assert "EQ" in types
        assert "INT" in types


class TestAgLSnippets:
    def test_let_declaration(self) -> None:
        result = tok("let x = 42")
        types = [t for t, _ in result]
        assert types == ["let", "NAME", "EQ", "INT"]

    def test_var_declaration(self) -> None:
        result = tok("var artifact: text = 1")
        types = [t for t, _ in result]
        assert "var" in types
        assert "COLON" in types

    def test_agent_call_expression(self) -> None:
        result = tok('reviewer "Review ${artifact}"')
        types = [t for t, _ in result]
        assert types[0] == "NAME"
        assert "TEMPLATE_START" in types
        assert "INTERP_START" in types

    def test_enum_declaration(self) -> None:
        # The |‑continuation rule makes the variant list flat (no _INDENT/_DEDENT).
        # Variants start with |, so the layout filter suppresses newlines before them.
        source = "enum Review\n  | Pass\n  | Fail"
        result = tok(source)
        types = [t for t, _ in result]
        assert "enum" in types
        assert "NAME" in types
        assert "PIPE" in types
        # No _INDENT/_DEDENT: | variants are flat after the |‑continuation rule
        assert "_INDENT" not in types
        assert "_DEDENT" not in types

    def test_do_until_loop(self) -> None:
        source = "do[5] pass until true"
        result = tok(source)
        types = [t for t, _ in result]
        assert "do" in types
        assert "LSQB" in types
        assert "INT" in types
        assert "RSQB" in types
        assert "until" in types
        assert "true" in types

    def test_if_branch_statement(self) -> None:
        source = "if x =>\n  pass\n| else =>\n  pass"
        result = tok(source)
        types = [t for t, _ in result]
        assert "if" in types
        assert "ARROW" in types
        assert "_INDENT" in types
        assert "PIPE" in types
        assert "else" in types
        assert "_DEDENT" in types

    def test_type_annotation(self) -> None:
        result = tok("let x: Review = foo")
        types = [t for t, _ in result]
        assert "COLON" in types
        assert "NAME" in types

    def test_field_access(self) -> None:
        result = tok("e.raw")
        assert result == [("NAME", "e"), ("DOT", "."), ("NAME", "raw")]

    def test_case_of_statement(self) -> None:
        # The |‑continuation rule makes the branch list flat (no _INDENT for branch header).
        # The branch body (pass) at the same level as | doesn't get its own indent.
        source = "case x of\n  | A => pass"
        result = tok(source)
        types = [t for t, _ in result]
        assert "case" in types
        assert "of" in types
        assert "PIPE" in types

    def test_try_catch_block(self) -> None:
        source = "try\n  pass\ncatch E as e =>\n  pass"
        result = tok(source)
        types = [t for t, _ in result]
        assert "try" in types
        assert "catch" in types
        assert "as" in types
        assert "ARROW" in types

    def test_multiline_template_with_interpolation(self) -> None:
        source = '"""\nhello\n${name}\nworld\n"""'
        result = tok(source)
        types = [t for t, _ in result]
        assert "TEMPLATE_START" in types
        assert "INTERP_START" in types
        assert "INTERP_END" in types
        assert "TEMPLATE_END" in types

    def test_arithmetic_expression(self) -> None:
        result = tok("x + 1 * y - z / 2")
        types = [t for t, _ in result]
        assert "PLUS" in types
        assert "STAR" in types
        assert "MINUS" in types
        assert "SLASH" in types

    def test_comparison_operators(self) -> None:
        result = tok("x <= y and a >= b")
        types = [t for t, _ in result]
        assert "LE" in types
        assert "and" in types
        assert "GE" in types

    def test_complex_program_fragment(self) -> None:
        source = """enum Review
  | Pass
  | Fail

let review: Review = reviewer "Review ${artifact}"

case review of
  | Pass => pass
  | Fail => pass"""
        result = tok(source)
        types = [t for t, _ in result]
        # Just ensure no crash and key tokens present
        assert "enum" in types
        assert "let" in types
        assert "case" in types
        assert types.count("_INDENT") == types.count("_DEDENT")

    def test_record_declaration(self) -> None:
        source = "record Issue\n  title: text\n  severity: int"
        result = tok(source)
        types = [t for t, _ in result]
        assert "record" in types
        assert "COLON" in types
        assert "_INDENT" in types
        assert "_DEDENT" in types

    def test_print_statement(self) -> None:
        # In AgL, `print` is an ordinary function name (VAR_NAME), not a reserved keyword.
        result = tok('print "hello"')
        types = [t for t, _ in result]
        assert types[0] == "NAME"
        assert result[0][1] == "print"
        assert "TEMPLATE_START" in types

    def test_raise_statement(self) -> None:
        result = tok("raise e")
        types = [t for t, _ in result]
        assert types[0] == "raise"
        assert "NAME" in types

    def test_not_and_or_keywords(self) -> None:
        result = tok("not true and false or null")
        types = [t for t, _ in result]
        assert "not" in types
        assert "and" in types
        assert "or" in types

    def test_is_in_keywords(self) -> None:
        result = tok("x is A and y in z")
        types = [t for t, _ in result]
        assert "is" in types
        assert "and" in types
        assert "in" in types


# ---------------------------------------------------------------------------
# CRLF / universal-newline normalization
# ---------------------------------------------------------------------------


class TestNewlineNormalization:
    def test_crlf_program_lexes_like_lf_twin(self) -> None:
        # A CRLF source must produce an identical (type, value) stream to its
        # LF twin -- newlines are normalized at scanner entry.
        lf = "if x =>\n  pass\nother"
        crlf = lf.replace("\n", "\r\n")
        assert tok(crlf) == tok(lf)

    def test_crlf_inside_triple_quoted_string(self) -> None:
        # CRLF inside a triple-quoted string is normalized to LF in the fragment.
        lf = '"""\n  hello\n  world\n  """'
        crlf = lf.replace("\n", "\r\n")
        lf_frags = "".join(v for t, v in tok(lf) if t == "STRING_FRAGMENT")
        crlf_frags = "".join(v for t, v in tok(crlf) if t == "STRING_FRAGMENT")
        assert crlf_frags == lf_frags
        assert "\r" not in crlf_frags

    def test_lone_cr_is_normalized(self) -> None:
        # A classic-Mac lone-CR source lexes identically to its LF twin.
        lf = "a\nb\nc"
        cr = "a\rb\rc"
        assert tok(cr) == tok(lf)

    def test_shared_normalize_newlines_helper(self) -> None:
        """the shared universal-newline helper converts CRLF and lone CR to
        LF (the single source of truth shared by the scanner and the evaluator)."""
        from agm.util.text import normalize_newlines

        assert normalize_newlines("a\r\nb\rc\nd") == "a\nb\nc\nd"
        assert normalize_newlines("") == ""
        # Idempotent on already-LF text.
        assert normalize_newlines("x\ny") == "x\ny"


# ---------------------------------------------------------------------------
# leading _NEWLINE suppression
# ---------------------------------------------------------------------------


class TestLeadingNewlineSuppression:
    def test_comment_first_file_no_leading_newline(self) -> None:
        result = tok("# header comment\nlet x = 1")
        types = [t for t, _ in result]
        assert types[0] == "let"
        assert types[0] != "_NEWLINE"

    def test_blank_first_file_no_leading_newline(self) -> None:
        result = tok("\nlet x = 1")
        types = [t for t, _ in result]
        assert types[0] == "let"

    def test_multiple_leading_blanks_no_leading_newline(self) -> None:
        result = tok("\n\n\n# c\n\nlet x = 1")
        types = [t for t, _ in result]
        assert types[0] == "let"
        # No _NEWLINE precedes the first real token.
        assert "_NEWLINE" not in types[: types.index("let")]

    def test_entirely_blank_or_comment_only_file(self) -> None:
        # A file with only blanks/comments yields no _NEWLINE and no layout noise.
        result = tok("\n\n# only a comment\n\n")
        types = [t for t, _ in result]
        assert "_NEWLINE" not in types
        assert "_INDENT" not in types
        assert "_DEDENT" not in types
        assert types == []


# ---------------------------------------------------------------------------
# targeted newline-inside-interpolation diagnostic
# ---------------------------------------------------------------------------


class TestNewlineInsideInterpolation:
    def test_newline_in_single_quoted_interp_is_rejected(self) -> None:
        with pytest.raises(LexError) as exc_info:
            tok('"${a\nb}"')
        err = exc_info.value
        assert err.span is not None
        assert "interpolation" in str(err)
        assert "newline" in str(err)

    def test_newline_in_triple_quoted_interp_is_rejected(self) -> None:
        with pytest.raises(LexError) as exc_info:
            tok('"""${a\nb}"""')
        err = exc_info.value
        assert err.span is not None
        assert "interpolation" in str(err)


# ---------------------------------------------------------------------------
# token position threading
# ---------------------------------------------------------------------------


class TestTripleTemplatePositions:
    def test_triple_template_tokens_have_positions_in_source_range(self) -> None:
        source = '"""\n  hello ${name}\n  world\n  """'
        tokens = [
            t
            for t in tokenize(source)
            if t.type in ("STRING_FRAGMENT", "INTERP_START", "TEMPLATE_END")
        ]
        assert tokens  # sanity
        for t in tokens:
            assert t.line is not None
            assert t.column is not None
            assert t.start_pos is not None
            assert t.end_pos is not None
            # Positions must point INTO the template's true source range.
            assert 0 <= t.start_pos <= len(source)
            assert 0 <= t.end_pos <= len(source)

    def test_triple_template_positions_monotonic_non_decreasing(self) -> None:
        source = '"""\n  a ${x} b ${y} c\n  """'
        starts = [
            t.start_pos
            for t in tokenize(source)
            if t.type in ("STRING_FRAGMENT", "INTERP_START", "INTERP_END", "TEMPLATE_END")
            and t.start_pos is not None
        ]
        assert starts == sorted(starts)

    def test_single_template_fragment_does_not_overlap_closing_quote(self) -> None:
        # Regression: STRING_FRAGMENT's end_pos used to span the closing quote
        # (it was emitted after consuming the quote), overlapping TEMPLATE_END
        # and duplicating the quote in span consumers like the REPL highlighter.
        for source in ('x = "hi"', "x = 'hi'", 'f("a" + "b")'):
            spanned = [
                t
                for t in tokenize(source)
                if t.start_pos is not None
                and t.end_pos is not None
                and t.end_pos > t.start_pos
            ]
            ends = sorted(spanned, key=lambda t: t.start_pos)
            for prev, nxt in zip(ends, ends[1:]):
                assert prev.end_pos <= nxt.start_pos, (
                    f"overlap in {source!r}: {prev.type} {prev.value!r} "
                    f"[{prev.start_pos},{prev.end_pos}) overlaps "
                    f"{nxt.type} {nxt.value!r} [{nxt.start_pos},{nxt.end_pos})"
                )
            frag = next(t for t in tokenize(source) if t.type == "STRING_FRAGMENT")
            assert frag.end_pos == frag.start_pos + len(frag.value)

    def test_layout_tokens_have_positions(self) -> None:
        # _NEWLINE/_INDENT/_DEDENT must all carry concrete positions.
        source = "a\n  b\nc"
        for t in tokenize(source):
            if t.type in ("_NEWLINE", "_INDENT", "_DEDENT"):
                assert t.start_pos is not None
                assert t.end_pos is not None
                assert t.line is not None
                assert t.column is not None

    def test_newline_token_positioned_at_newline_char(self) -> None:
        # Per the layout rule, a _NEWLINE sits at the newline character itself.
        source = "a\nb"
        nl = next(t for t in tokenize(source) if t.type == "_NEWLINE")
        assert source[nl.start_pos] == "\n"


# ---------------------------------------------------------------------------
# Bug regression: triple-quoted dedent placeholder collision
# ---------------------------------------------------------------------------


class TestTripleDedentPlaceholderCollision:
    """The in-band placeholder \x00INTERP\x00 must never collide with real string
    content — a triple-quoted literal whose decoded text contains that exact
    byte sequence (constructible via \\u0000, \\u0049, etc.) must lex correctly,
    not crash with an IndexError."""

    def test_null_byte_in_triple_string_no_crash(self) -> None:
        # \u0000 produces the NUL byte, which is part of the old placeholder.
        # The string must lex to a single STRING_FRAGMENT containing the NUL.
        source = '"""\n\\u0000\n"""'
        result = tok(source)
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        assert "".join(frags) == "\x00"

    def test_placeholder_sequence_in_triple_string_no_crash(self) -> None:
        # Construct the exact old placeholder "\x00INTERP\x00" via escapes:
        # \u0000 = NUL, then literal "INTERP", then \u0000 again.
        # This should lex cleanly — not crash with IndexError.
        source = '"""\n\\u0000INTERP\\u0000\n"""'
        result = tok(source)
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        assert "".join(frags) == "\x00INTERP\x00"

    def test_placeholder_in_triple_string_with_interpolation_no_crash(self) -> None:
        # Same placeholder embedded alongside a real interpolation.
        # With the old split-on-sentinel approach this produces too many parts
        # and crashes on lit_segs[part_idx] — must not crash.
        source = '"""\n\\u0000INTERP\\u0000 ${x}\n"""'
        result = tok(source)
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        interp_starts = [(t, v) for t, v in result if t == "INTERP_START"]
        # First fragment should contain the placeholder text followed by a space
        assert frags[0] == "\x00INTERP\x00 "
        assert len(interp_starts) == 1

    def test_multiple_placeholder_sequences_with_interp_no_crash(self) -> None:
        # Two placeholder sequences, one interpolation.  The old code split on
        # "\x00INTERP\x00" and got 3 literal parts for 1 literal segment -> crash.
        source = '"""\\u0000INTERP\\u0000${x}\\u0000INTERP\\u0000"""'
        result = tok(source)
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        assert frags[0] == "\x00INTERP\x00"
        assert frags[1] == "\x00INTERP\x00"


# ---------------------------------------------------------------------------
# Identifiers: Unicode and symbol characters
# ---------------------------------------------------------------------------


class TestIdentifierUnicodeAndSymbols:
    """Identifiers start with a (Unicode) letter or ``_`` and then greedily
    consume every character that is not whitespace and not a structural
    operator/punctuator delimiter.  Non-ASCII letters, digits, and the symbol
    characters ``-``, ``?``, ``!`` are all valid identifier-continuation
    characters, so names like ``ask-prompt`` or ``do-it-now!`` scan as a single
    token.  Operator tokens (``->``, ``=>``, ``!=``, field access ``.``, etc.)
    still lex as operators when they appear as standalone, whitespace-delimited
    tokens."""

    def test_latin_extended_lowercase_accepted(self) -> None:
        # é (U+00E9) is a Unicode letter — a valid NAME start.
        assert tok("é") == [("NAME", "é")]

    def test_greek_uppercase_accepted(self) -> None:
        # Ω (U+03A9) — identifiers are case-neutral, so this is a NAME.
        assert tok("Ω") == [("NAME", "Ω")]

    def test_cjk_identifier_accepted(self) -> None:
        assert tok("中文变量") == [("NAME", "中文变量")]

    def test_ascii_lowercase_identifier_still_valid(self) -> None:
        assert tok("foo_bar") == [("NAME", "foo_bar")]

    def test_ascii_uppercase_identifier_still_valid(self) -> None:
        assert tok("FooBar") == [("NAME", "FooBar")]

    def test_identifier_with_digits_still_valid(self) -> None:
        assert tok("x1") == [("NAME", "x1")]

    def test_underscore_only_identifier_still_valid(self) -> None:
        assert tok("_x") == [("NAME", "_x")]

    def test_unicode_letter_in_identifier_continuation_accepted(self) -> None:
        # "aé" — starts with ASCII 'a', continues with Unicode 'é'.
        assert tok("aé") == [("NAME", "aé")]

    def test_hyphen_in_identifier(self) -> None:
        assert tok("ask-prompt") == [("NAME", "ask-prompt")]

    def test_question_mark_in_identifier(self) -> None:
        assert tok("ask?") == [("NAME", "ask?")]

    def test_exclamation_in_identifier(self) -> None:
        assert tok("do-it-now!") == [("NAME", "do-it-now!")]

    def test_operator_chars_in_identifier(self) -> None:
        # -, >, +, * (but NOT =) are identifier-continuation characters, so
        # a run of them with no surrounding whitespace stays one identifier.
        assert tok("a->b") == [("NAME", "a->b")]
        assert tok("a+b") == [("NAME", "a+b")]
        assert tok("n*x") == [("NAME", "n*x")]
        # = is a stop character: a=b lexes as NAME EQ NAME, not a single identifier.
        assert tok("x!=3") == [("NAME", "x!"), ("EQ", "="), ("INT", "3")]
        assert tok("Pass=>()") == [("NAME", "Pass"), ("ARROW", "=>"), ("LPAR", "("), ("RPAR", ")")]

    def test_spaces_break_identifier_before_operator(self) -> None:
        # Whitespace is a stop character, so the operator tokens re-emerge.
        assert tok("a - b") == [
            ("NAME", "a"), ("MINUS", "-"), ("NAME", "b"),
        ]
        assert tok("a -> b") == [
            ("NAME", "a"), ("THIN_ARROW", "->"), ("NAME", "b"),
        ]
        assert tok("x != 3") == [
            ("NAME", "x"), ("NEQ", "!="), ("INT", "3"),
        ]
        assert tok("a + b") == [
            ("NAME", "a"), ("PLUS", "+"), ("NAME", "b"),
        ]
        assert tok("a * b") == [
            ("NAME", "a"), ("STAR", "*"), ("NAME", "b"),
        ]
        assert tok("Pass => ()") == [
            ("NAME", "Pass"), ("ARROW", "=>"), ("LPAR", "("), ("RPAR", ")"),
        ]

    def test_string_quotes_inside_identifier(self) -> None:
        # Both " and ' are identifier-continuation characters, not delimiters.
        assert tok('foo"bar') == [("NAME", 'foo"bar')]
        assert tok("foo'bar") == [("NAME", "foo'bar")]
        # A leading quote (or one after whitespace) still starts a template.
        assert tok('"hello"') == [
            ("TEMPLATE_START", '"'), ("STRING_FRAGMENT", "hello"),
            ("TEMPLATE_END", '"'),
        ]
        assert tok("ask 'x'") == [
            ("NAME", "ask"),
            ("TEMPLATE_START", "'"), ("STRING_FRAGMENT", "x"), ("TEMPLATE_END", "'"),
        ]

    def test_structural_punctuators_break_identifier(self) -> None:
        # `.` `:` `,` `(` `)` etc. are stop characters even without spaces.
        assert tok("a.b") == [("NAME", "a"), ("DOT", "."), ("NAME", "b")]
        assert tok("f(x)") == [
            ("NAME", "f"), ("LPAR", "("), ("NAME", "x"), ("RPAR", ")"),
        ]
        assert tok("a:b") == [("NAME", "a"), ("COLON", ":"), ("NAME", "b")]

    def test_hyphen_cannot_start_identifier(self) -> None:
        # An identifier must start with a letter or _; a leading '-' is the
        # MINUS / THIN_ARROW operator path.
        assert tok("-a") == [("MINUS", "-"), ("NAME", "a")]

    def test_digit_cannot_start_identifier(self) -> None:
        # A leading digit begins a number, not an identifier.
        assert tok("2") == [("INT", "2")]

    def test_non_ascii_digit_rejected_as_number(self) -> None:
        # Fullwidth and Arabic-Indic digits are not ASCII [0-9]; the language
        # defines numeric literals as [0-9]+, so these are unexpected characters
        # rather than INT tokens.
        for ch in ("２", "٠"):
            with pytest.raises(LexError) as exc_info:
                tok(ch)
            assert exc_info.value.span is not None


# ---------------------------------------------------------------------------
# Bug regression: triple-quoted dedent over-strips when interpolation hole
# is on the minimum-indented line
# ---------------------------------------------------------------------------


class TestTripleDedentHoleAwareMinIndent:
    """The dedent rule must treat a line containing only an interpolation hole
    (e.g. ``  ${x}``) as non-blank when computing min-indent.  Previously,
    holes were stripped from the combined literal before computing min-indent,
    turning ``  ${x}`` into pure whitespace ``  `` which was classified as a
    blank line and excluded.  This caused the wrong (larger) indent to be
    stripped, over-removing indentation from other content lines.

    Design rule: a line with a hole IS non-blank; the indent of that
    line counts toward the minimum.  Hole values are never dedented.
    """

    def test_hole_line_is_unique_minimum_indent(self) -> None:
        # Source (indent shown as explicit spaces):
        #   """
        #       deep    <- indent 6
        #   ${x}        <- indent 2 (the true minimum)
        #   """         <- closing at indent 2 (same as hole)
        #
        # Correct dedent: strip 2.  "deep" ends up with 4 leading spaces.
        # Buggy dedent: strip 6.  "deep" ends up with 0 leading spaces.
        source = '"""\n      deep\n  ${x}\n  """'
        result = tok(source)
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        # Fragment before ${x}: "    deep\n" (4 leading spaces kept after strip 2)
        assert frags[0] == "    deep\n"
        # Fragment after ${x}: "" — the closing "  """ segment strips to nothing
        # ("\n  " → "\n" after strip 2 → trailing \n removed → "").
        assert frags[1] == ""

    def test_hole_only_line_between_content_lines(self) -> None:
        # Source:
        #   """
        #     content   <- indent 4
        #   ${x}        <- indent 2 (minimum)
        #     more      <- indent 4
        #   """         <- closing at indent 2 (same as hole)
        #
        # Correct dedent: strip 2.  Both content lines keep 2 leading spaces.
        source = '"""\n    content\n  ${x}\n    more\n  """'
        result = tok(source)
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        assert frags[0] == "  content\n"
        # Fragment after ${x}: "\n  more" — newline + 2 spaces before "more"
        assert frags[1] == "\n  more"

    def test_hole_at_line_start_with_trailing_literal_text(self) -> None:
        # Source:
        #   """
        #     prefix    <- indent 4
        #   ${x}tail    <- indent 2 (minimum, hole at col 2, literal "tail" follows)
        #   """         <- closing at indent 2
        #
        # Correct dedent: strip 2.
        source = '"""\n    prefix\n  ${x}tail\n  """'
        result = tok(source)
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        # Before the hole: "  prefix\n" (4-2=2 leading spaces preserved)
        assert frags[0] == "  prefix\n"
        # After the hole on the same line: "tail\n" then closing line strips away
        # "\n  " → after strip 2 → "\n" → trailing \n removed → so "tail"
        assert frags[1] == "tail"

    def test_existing_dedent_no_holes_unchanged(self) -> None:
        # Regression guard: no-hole case must still work correctly.
        source = '"""\n    hello\n      world\n    """'
        result = tok(source)
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        content = "".join(frags)
        assert content == "hello\n  world"

    def test_sentinel_collision_still_passes(self) -> None:
        # Literal \x00INTERP\x00 content alongside a real interpolation must
        # still work (verifies the old sentinel-collision regression fix).
        source = '"""\n\\u0000INTERP\\u0000 ${x}\n"""'
        result = tok(source)
        frags = [v for t, v in result if t == "STRING_FRAGMENT"]
        assert frags[0] == "\x00INTERP\x00 "
        assert len([v for t, v in result if t == "INTERP_START"]) == 1


# ---------------------------------------------------------------------------
# TAB character warnings
# ---------------------------------------------------------------------------


class TestTabWarnings:
    def test_empty_source_no_warnings(self) -> None:
        assert lex_tab_warnings("") == []

    def test_source_without_tabs_no_warnings(self) -> None:
        assert lex_tab_warnings("let x = 1\n    y = 2") == []

    def test_single_tab_yields_one_warning(self) -> None:
        warnings = lex_tab_warnings("\tlet x = 1")
        assert len(warnings) == 1
        assert warnings[0].severity == "warning"
        assert warnings[0].line == 1

    def test_warning_message_mentions_tab(self) -> None:
        warnings = lex_tab_warnings("let x =\t1")
        assert len(warnings) == 1
        assert "TAB" in warnings[0].message or "tab" in warnings[0].message.lower()

    def test_warning_includes_column_info(self) -> None:
        # Tab is the 9th character (column 9) on the line.
        warnings = lex_tab_warnings("let x = \t1")
        assert len(warnings) == 1
        assert "9" in warnings[0].message

    def test_tab_on_second_line_reported_at_correct_line(self) -> None:
        warnings = lex_tab_warnings("let x = 1\n\tlet y = 2")
        assert len(warnings) == 1
        assert warnings[0].line == 2

    def test_tabs_on_multiple_lines_each_get_warning(self) -> None:
        warnings = lex_tab_warnings("\tlet x = 1\n\tlet y = 2")
        assert len(warnings) == 2
        assert warnings[0].line == 1
        assert warnings[1].line == 2

    def test_multiple_tabs_on_same_line_each_get_warning(self) -> None:
        warnings = lex_tab_warnings("let\tx\t= 1")
        assert len(warnings) == 2
        lines = [w.line for w in warnings]
        assert all(ln == 1 for ln in lines)

    def test_crlf_newlines_normalized_correctly(self) -> None:
        # CRLF: tab on second line should still be line 2.
        warnings = lex_tab_warnings("let x = 1\r\n\tlet y = 2")
        assert len(warnings) == 1
        assert warnings[0].line == 2

    def test_cr_newlines_normalized_correctly(self) -> None:
        warnings = lex_tab_warnings("let x = 1\r\tlet y = 2")
        assert len(warnings) == 1
        assert warnings[0].line == 2

    def test_tab_in_string_literal_is_allowed(self) -> None:
        # A literal TAB inside string content is allowed (only code TABs are
        # advised against), so it produces no warning.
        assert lex_tab_warnings('let x = "hello\tworld"') == []

    def test_tab_in_triple_quoted_string_is_allowed(self) -> None:
        # Same for triple-quoted templates: TABs in the literal body are fine.
        assert lex_tab_warnings('let x = """\n\tindented body\n"""') == []

    def test_tab_in_interpolation_is_code_and_warned(self) -> None:
        # An interpolation hole is CODE, so a TAB inside ``${ ... }`` warns even
        # though it sits within a string literal.
        warnings = lex_tab_warnings('let y = 1\nlet x = "${y\t}"')
        assert len(warnings) == 1
        assert warnings[0].line == 2

    def test_tab_in_comment_only_line_still_warned(self) -> None:
        # A TAB inside a comment-only line (skipped during indentation
        # measurement) is still reported, on the comment's line.
        warnings = lex_tab_warnings("let x = 1\n#\tcomment\nlet y = 2")
        assert len(warnings) == 1
        assert warnings[0].line == 2

    def test_tab_in_trailing_whitespace_only_line_warned_once(self) -> None:
        # A whitespace-only EOF tail containing a TAB is reported exactly once
        # (the indentation probe rolls back so the code-mode rescan does not
        # double-count it).
        warnings = lex_tab_warnings("let x = 1\n\t")
        assert len(warnings) == 1
        assert warnings[0].line == 2

    def test_warnings_are_diagnostic_instances(self) -> None:
        warnings = lex_tab_warnings("\tx = 1")
        assert all(isinstance(w, Diagnostic) for w in warnings)
        assert all(w.severity == "warning" for w in warnings)


# ---------------------------------------------------------------------------
# AgL token changes
# ---------------------------------------------------------------------------


class TestV2Keywords:
    """Tests for new and changed keyword reservation in AgL."""

    # --- def is a new reserved keyword ---

    def test_def_lexes_as_keyword_token(self) -> None:
        result = tok("def")
        assert result == [("def", "def")]

    def test_def_is_reserved_not_var_name(self) -> None:
        types = [t for t, _ in tok("def")]
        assert "NAME" not in types
        assert "def" in types

    def test_def_prefix_identifier_is_var_name(self) -> None:
        # "default" starts with "def" but must lex as a plain identifier.
        result = tok("default")
        assert result == [("NAME", "default")]

    def test_def_remapped_to_uppercase_in_lark_interface(self) -> None:
        # AglLexer.lex() remaps lowercase "def" → "DEF" for the grammar.
        lexer = AglLexer(None)
        state = LexerState("def")
        result = list(lexer.lex(state, None))
        assert len(result) == 1
        assert result[0].type == "DEF"

    # --- fn is a new reserved keyword ---

    def test_fn_lexes_as_keyword_token(self) -> None:
        result = tok("fn")
        assert result == [("fn", "fn")]

    def test_fn_is_reserved_not_var_name(self) -> None:
        types = [t for t, _ in tok("fn")]
        assert "NAME" not in types
        assert "fn" in types

    def test_fn_prefix_identifier_is_var_name(self) -> None:
        # "find" starts with "fn"? No — "fn" would not prefix "find" anyway.
        # But "fnord" starts with "fn" and must be a plain identifier.
        result = tok("fnord")
        assert result == [("NAME", "fnord")]

    def test_fn_remapped_to_uppercase_in_lark_interface(self) -> None:
        lexer = AglLexer(None)
        state = LexerState("fn")
        result = list(lexer.lex(state, None))
        assert len(result) == 1
        assert result[0].type == "FN"

    # --- pass is no longer a reserved keyword ( role taken by ()) ---

    def test_pass_lexes_as_var_name(self) -> None:
        result = tok("pass")
        assert result == [("NAME", "pass")]

    def test_pass_not_a_keyword_type(self) -> None:
        types = [t for t, _ in tok("pass")]
        assert "pass" not in types
        assert "NAME" in types

    # --- print is no longer a reserved keyword ( ordinary function name) ---

    def test_print_lexes_as_var_name(self) -> None:
        result = tok("print")
        assert result == [("NAME", "print")]

    def test_print_not_a_keyword_type(self) -> None:
        types = [t for t, _ in tok("print")]
        assert "print" not in types
        assert "NAME" in types

    def test_print_used_as_call_lexes_correctly(self) -> None:
        # print(x) should now lex as: NAME LPAR NAME RPAR
        result = tok("print(x)")
        types = [t for t, _ in result]
        assert types == ["NAME", "LPAR", "NAME", "RPAR"]
        assert result[0] == ("NAME", "print")

    # --- unit is non-reserved (like text/int/bool — just a NAME) ---

    def test_unit_lexes_as_var_name(self) -> None:
        # `unit` is a type keyword but not reserved — it lexes as NAME,
        # exactly like `text`, `int`, `bool`, `json`, `decimal`, `list`, `dict`.
        result = tok("unit")
        assert result == [("NAME", "unit")]

    def test_unit_usable_as_type_annotation_context(self) -> None:
        # `unit` in a type annotation position lexes as NAME (grammar
        # handles it as a primitive type — same as "text", "int", etc.).
        result = tok("x: unit")
        types = [t for t, _ in result]
        assert types == ["NAME", "COLON", "NAME"]
        values = [v for _, v in result]
        assert values[-1] == "unit"

    def test_unit_usable_as_identifier_no_crash(self) -> None:
        # Since unit is non-reserved, it can appear as an identifier too.
        result = tok("let unit = 1")
        types = [t for t, _ in result]
        assert "let" in types
        assert "NAME" in types

    def test_existing_primitive_type_keywords_still_lex_as_var_name(self) -> None:
        # Regression guard: text, int, bool, json, decimal, list, dict
        # must all still be NAME (unchanged from before).
        for word in ("text", "int", "bool", "json", "decimal", "list", "dict"):
            result = tok(word)
            assert result == [("NAME", word)], f"{word!r} should be NAME"

    # --- mixed current keywords in a snippet ---

    def test_def_and_fn_reserved_together(self) -> None:
        result = tok("def fn")
        types = [t for t, _ in result]
        assert types == ["def", "fn"]

    def test_still_reserved_keywords_unchanged(self) -> None:
        # let/var/do/until/if/else/case/of/try/catch/raise/as/and/or/not
        # are still reserved.
        source = "let var set do until if else case of try catch raise as and or not"
        result = tok(source)
        types = [t for t, _ in result]
        for kw in ("let", "var", "do", "until", "if", "else", "case",
                   "of", "try", "catch", "raise", "as", "and", "or", "not"):
            assert kw in types, f"keyword {kw!r} must still be reserved"


class TestV2ThinArrow:
    """Tests for the new THIN_ARROW (->) token, distinct from ARROW (=>)."""

    def test_thin_arrow_emits_thin_arrow_token(self) -> None:
        result = tok("->")
        assert result == [("THIN_ARROW", "->")]

    def test_fat_arrow_still_emits_arrow_token(self) -> None:
        # => must still be ARROW (unchanged)
        result = tok("=>")
        assert result == [("ARROW", "=>")]

    def test_thin_arrow_not_confused_with_minus_gt(self) -> None:
        # -> as a single token, not MINUS then GT
        result = tok("->")
        assert len(result) == 1
        assert result[0][0] == "THIN_ARROW"

    def test_minus_then_var_is_minus_and_var(self) -> None:
        # "- x" must still be MINUS NAME (not consumed as THIN_ARROW)
        result = tok("- x")
        types = [t for t, _ in result]
        assert types == ["MINUS", "NAME"]

    def test_minus_not_followed_by_gt_is_minus(self) -> None:
        # "-1" should still be MINUS INT
        result = tok("- 1")
        types = [t for t, _ in result]
        assert types == ["MINUS", "INT"]

    def test_gt_alone_is_gt(self) -> None:
        # standalone ">" must still be GT
        result = tok(">")
        assert result == [("GT", ">")]

    def test_ge_still_works(self) -> None:
        # >= must still be GE
        result = tok(">=")
        assert result == [("GE", ">=")]

    def test_func_type_annotation_lexes_correctly(self) -> None:
        # "(int) -> text" as a type expression lexes with THIN_ARROW
        result = tok("(int) -> text")
        types = [t for t, _ in result]
        assert "LPAR" in types
        assert "NAME" in types  # int
        assert "RPAR" in types
        assert "THIN_ARROW" in types
        assert "NAME" in types  # text

    def test_thin_arrow_in_def_signature_context(self) -> None:
        # def f(x: int) -> text = body
        result = tok("def f(x: int) -> text")
        types = [t for t, _ in result]
        assert "def" in types
        assert "THIN_ARROW" in types

    def test_lambda_with_thin_arrow_return_type(self) -> None:
        # fn(x: int) -> int => x
        result = tok("fn(x: int) -> int => x")
        types = [t for t, _ in result]
        assert "fn" in types
        assert "THIN_ARROW" in types
        assert "ARROW" in types

    def test_thin_arrow_remapped_to_uppercase_in_lark_interface(self) -> None:
        # THIN_ARROW must appear in the Lark interface stream unchanged
        # (it's not a keyword remap, it's a punctuation token — already uppercase).
        lexer = AglLexer(None)
        state = LexerState("->")
        result = list(lexer.lex(state, None))
        assert len(result) == 1
        assert result[0].type == "THIN_ARROW"

    def test_maximal_munch_thin_arrow_over_minus_gt(self) -> None:
        # "->" must be one THIN_ARROW, not MINUS + GT
        result = tok("->")
        assert result == [("THIN_ARROW", "->")]

    def test_arrow_vs_thin_arrow_in_sequence(self) -> None:
        # Both tokens in the same snippet
        result = tok("-> =>")
        assert result == [("THIN_ARROW", "->"), ("ARROW", "=>")]


class TestV2LoopBoundPreserved:
    """Verify the do[N] bracket tokens are preserved for the grammar."""

    def _lex(self, source: str) -> list[tuple[str, str]]:
        """Lex via AglLexer (Lark interface)."""
        lexer = AglLexer(None)
        state = LexerState(source)
        return [(t.type, str(t)) for t in lexer.lex(state, None)]

    def test_do_loop_bound_merge_preserved(self) -> None:
        result = self._lex("do[5]")
        types = [t for t, _ in result]
        assert types == ["DO", "DO_LSQB", "INT", "RSQB"]

    def test_loop_bound_value_is_integer_string(self) -> None:
        result = self._lex("do[10]")
        lb = next(v for t, v in result if t == "INT")
        assert lb == "10"

    def test_loop_bound_not_confused_with_list_literal(self) -> None:
        # [5] alone (not after do) must NOT become LOOP_BOUND even via AglLexer.
        result = self._lex("[5]")
        types = [t for t, _ in result]
        assert "LSQB" in types
        assert "INT" in types
        assert "RSQB" in types

    def test_do_with_body_and_thin_arrow_no_conflict(self) -> None:
        # Ensure the thin-arrow addition doesn't interfere with do[N] tokenization.
        result = self._lex("do[3] x -> y")
        types = [t for t, _ in result]
        assert types[:4] == ["DO", "DO_LSQB", "INT", "RSQB"]
        assert "THIN_ARROW" in types


# ---------------------------------------------------------------------------
# Case-neutral identifiers
# ---------------------------------------------------------------------------


class TestCaseNeutralNames:
    """All identifiers lex as NAME regardless of case."""

    def test_box_uppercase_is_name(self) -> None:
        result = tok("Box")
        assert result == [("NAME", "Box")]

    def test_foo_lowercase_is_name(self) -> None:
        result = tok("foo")
        assert result == [("NAME", "foo")]

    def test_underscore_x_is_name(self) -> None:
        result = tok("_x")
        assert result == [("NAME", "_x")]

    def test_box_lowercase_is_name(self) -> None:
        result = tok("box")
        assert result == [("NAME", "box")]

    def test_foo_uppercase_is_name(self) -> None:
        result = tok("Foo")
        assert result == [("NAME", "Foo")]

    def test_keywords_still_produce_keyword_tokens(self) -> None:
        result = tok("let var def fn")
        types = [t for t, _ in result]
        assert "let" in types
        assert "var" in types
        assert "def" in types
        assert "fn" in types
        assert "NAME" not in types

    def test_config_is_keyword(self) -> None:
        result = tok("config")
        assert result == [("config", "config")]

    def test_agent_is_keyword_not_name(self) -> None:
        result = tok("agent")
        types = [t for t, _ in result]
        assert "NAME" not in types


# ---------------------------------------------------------------------------
# as? keyword tests
# ---------------------------------------------------------------------------


class TestAsQuestionKeyword:
    """Tests for the `as?` convertibility-test keyword token."""

    def test_as_question_is_keyword(self) -> None:
        # `as?` is a reserved keyword; raw scanner emits it as type "as?"
        result = tok("as?")
        assert result == [("as?", "as?")]

    def test_as_remains_keyword(self) -> None:
        # `as` alone stays as its existing keyword token type
        result = tok("as")
        assert result == [("as", "as")]

    def test_as_question_parser_token(self) -> None:
        # The parser-facing lex() remaps "as?" to "AS_QUESTION"
        result = lark_tok("as?")
        assert result == [("AS_QUESTION", "as?")]

    def test_as_parser_token(self) -> None:
        # The parser-facing lex() remaps "as" to "AS"
        result = lark_tok("as")
        assert result == [("AS", "as")]

    def test_ask_question_stays_name(self) -> None:
        # `ask?` is NOT a keyword — it stays NAME
        result = tok("ask?")
        assert result == [("NAME", "ask?")]

    def test_other_question_suffix_stays_name(self) -> None:
        # Other ?-suffixed identifiers are not keywords
        for word in ["done?", "valid?", "empty?"]:
            result = tok(word)
            assert result == [("NAME", word)], f"expected NAME for {word!r}"

    def test_as_space_question_is_not_as_question(self) -> None:
        # `as ?` (spaced) is NOT the test operator: only the contiguous lexeme
        # `as?` is the keyword. With a space, `as` scans as the keyword and `?`
        # scans as a separate operator name.
        assert tok("as ?") == [("as", "as"), ("OP_NAME", "?")]
        # And `as` followed by an ordinary word stays the keyword, not `as?`.
        assert lark_tok("as x") == [("AS", "as"), ("NAME", "x")]

    def test_as_question_in_context(self) -> None:
        # In a real expression context
        result = tok("x as? int")
        assert ("as?", "as?") in result
        assert ("NAME", "x") in result
        assert ("NAME", "int") in result

    def test_as_in_context(self) -> None:
        # `as` in a cast context
        result = tok("x as text")
        assert ("as", "as") in result


# ---------------------------------------------------------------------------
# Module system lexer tests
# ---------------------------------------------------------------------------


class TestModuleSystemLexer:
    """Tests for soft-keyword promotion and MODQUAL merging."""

    # --- import soft keyword ---

    def test_import_at_line_start_is_import_token(self) -> None:
        result = tok("import foo.bar")
        assert result[0] == ("IMPORT", "import")

    def test_import_mid_expression_stays_name(self) -> None:
        # 'import' after an operator is NOT at item-start → stays NAME
        result = tok("x + import")
        types = [t for t, _ in result]
        assert "IMPORT" not in types
        assert ("NAME", "import") in result

    def test_private_at_line_start_is_private_token(self) -> None:
        result = tok("private def f() -> text = x")
        assert result[0] == ("PRIVATE", "private")

    def test_private_not_at_line_start_stays_name(self) -> None:
        result = tok("x + private")
        assert ("NAME", "private") in result
        types = [t for t, _ in result]
        assert "PRIVATE" not in types

    def test_qualified_in_import_line(self) -> None:
        result = tok("import foo qualified")
        types = [t for t, _ in result]
        assert "QUALIFIED" in types

    def test_qualified_outside_import_line_stays_name(self) -> None:
        result = tok("let qualified = 1")
        assert ("NAME", "qualified") in result
        types = [t for t, _ in result]
        assert "QUALIFIED" not in types

    def test_using_in_import_line(self) -> None:
        result = tok("import foo using bar")
        types = [t for t, _ in result]
        assert "USING" in types

    def test_hiding_in_import_line(self) -> None:
        result = tok("import foo hiding bar")
        types = [t for t, _ in result]
        assert "HIDING" in types

    def test_using_outside_import_stays_name(self) -> None:
        result = tok("let using = 1")
        assert ("NAME", "using") in result

    def test_hiding_outside_import_stays_name(self) -> None:
        result = tok("let hiding = 1")
        assert ("NAME", "hiding") in result

    def test_import_window_closes_at_newline(self) -> None:
        # After newline, a new import line resets; 'using' on a different line
        # from 'import' is not inside the import window
        src = "import foo\nusing bar"
        result = tok(src)
        # 'using' here is on its own line not preceded by import on same line
        assert ("NAME", "using") in result

    # --- MODQUAL merging ---

    def test_simple_name_dcolon_name_merges_to_modqual(self) -> None:
        result = lark_tok("foo::bar")
        assert result[0] == ("MODQUAL", "foo")
        assert result[1][0] == "NAME"
        assert result[1][1] == "bar"

    def test_dotted_path_dcolon_merges_to_modqual(self) -> None:
        result = lark_tok("foo.bar::baz")
        assert result[0] == ("MODQUAL", "foo.bar")
        assert result[1][1] == "baz"

    def test_upper_name_dcolon_merges_to_modqual(self) -> None:
        result = lark_tok("Foo::Bar")
        assert result[0] == ("MODQUAL", "Foo")
        assert result[1][0] == "NAME"

    def test_typed_call_dcolon_lsqb_not_merged(self) -> None:
        # NAME DCOLON LSQB — must NOT be merged to MODQUAL (typed call syntax)
        result = lark_tok("foo::[int]()")
        types = [t for t, _ in result]
        assert "MODQUAL" not in types
        assert "DCOLON" in types

    def test_bare_dcolon_not_merged(self) -> None:
        # Leading :: (self-reference) — no preceding name, no merge
        result = lark_tok("::foo")
        types = [t for t, _ in result]
        assert "MODQUAL" not in types
        assert "DCOLON" in types

    def test_import_in_lark_token_stream(self) -> None:
        # After remap, 'import' at item-start becomes IMPORT in the parser stream
        result = lark_tok("import foo")
        assert result[0] == ("IMPORT", "import")


# ---------------------------------------------------------------------------
# AT token (@) — parameter-marker prefix for @pos / @std / @named
# ---------------------------------------------------------------------------


class TestAtToken:
    """Tests for the standalone AT token (``@``).

    ``@`` lexes as a single ``AT`` token.  A following identifier is emitted
    as a separate ``NAME`` token, so ``@pos``, ``@std``, and ``@named`` each
    produce two tokens: ``AT`` then ``NAME``.  ``pos``, ``std``, and ``named``
    without a leading ``@`` remain ordinary ``NAME`` tokens.
    """

    def test_at_alone_is_at_token(self) -> None:
        assert tok("@") == [("AT", "@")]

    def test_at_pos_is_at_name(self) -> None:
        assert tok("@pos") == [("AT", "@"), ("NAME", "pos")]

    def test_at_std_is_at_name(self) -> None:
        assert tok("@std") == [("AT", "@"), ("NAME", "std")]

    def test_at_named_is_at_name(self) -> None:
        assert tok("@named") == [("AT", "@"), ("NAME", "named")]

    def test_pos_without_at_is_name(self) -> None:
        # `pos` is not a keyword — it must remain a plain identifier everywhere.
        assert tok("pos") == [("NAME", "pos")]

    def test_std_without_at_is_name(self) -> None:
        assert tok("std") == [("NAME", "std")]

    def test_named_without_at_is_name(self) -> None:
        assert tok("named") == [("NAME", "named")]

    def test_at_adjacent_to_punctuation(self) -> None:
        # (@named, x) → LPAR AT NAME("named") COMMA NAME("x") RPAR
        assert tok("(@named, x)") == [
            ("LPAR", "("),
            ("AT", "@"),
            ("NAME", "named"),
            ("COMMA", ","),
            ("NAME", "x"),
            ("RPAR", ")"),
        ]

    def test_at_adjacent_to_operators(self) -> None:
        # Spaced @std between other tokens
        assert tok("/ @std *") == [
            ("SLASH", "/"),
            ("AT", "@"),
            ("NAME", "std"),
            ("STAR", "*"),
        ]

    def test_at_breaks_preceding_identifier(self) -> None:
        # `@` is in _IDENT_STOP, so it terminates a preceding identifier scan.
        assert tok("x@pos") == [("NAME", "x"), ("AT", "@"), ("NAME", "pos")]

    def test_at_with_space_before_name(self) -> None:
        # A space between @ and the name still yields exactly AT then NAME.
        assert tok("@ pos") == [("AT", "@"), ("NAME", "pos")]

    def test_at_token_position(self) -> None:
        # Verify the AT token carries correct position information.
        tokens = list(tokenize("@pos"))
        at_tok = tokens[0]
        assert at_tok.type == "AT"
        assert at_tok.line == 1
        assert at_tok.column == 1
        name_tok = tokens[1]
        assert name_tok.type == "NAME"
        assert str(name_tok) == "pos"
        assert name_tok.column == 2  # `pos` starts immediately after `@`
