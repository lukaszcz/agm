"""AglLexer: Lark custom lexer for the AgL DSL.

Composes :mod:`~agm.agl.lexer.scanner` (raw tokens) with
:mod:`~agm.agl.lexer.layout` (INDENT/DEDENT injection) to produce the full
token stream required by the Lark LALR parser.

The lexer is wired as::

    Lark(grammar, parser="lalr", lexer=AglLexer, propagate_positions=True)

The ``lex`` method receives ``lexer_state`` and ``parser_state`` from Lark.
We read the source text from ``lexer_state.text`` and yield
:class:`lark.lexer.Token` objects with full position information.

Grammar terminal names
----------------------
Lark creates uppercase terminal names from grammar string literals (e.g.
``"pass"`` → terminal ``PASS``).  Because the raw scanner emits lowercase
keyword types (``"pass"``, ``"let"``, etc.), ``lex()`` applies a one-to-one
remapping: ``GRAMMAR_TOKEN_REMAP`` maps each lowercase keyword type to its
uppercase equivalent so the LALR parser can look up the right parse-table
entry.

The public :func:`~agm.agl.lexer.tokenize` helper goes through
``layout(scan())`` directly and preserves the lowercase types that tests
and downstream tooling rely on.
"""

from __future__ import annotations

from typing import Iterator

from lark.lexer import Lexer, LexerState, Token

from agm.agl.lexer.layout import layout
from agm.agl.lexer.scanner import scan
from agm.agl.lexer.tokens import GRAMMAR_TOKEN_REMAP


def _remap(tokens: Iterator[Token]) -> Iterator[Token]:
    """Remap lowercase keyword token types to the uppercase names the grammar expects."""
    for tok in tokens:
        mapped = GRAMMAR_TOKEN_REMAP.get(tok.type)
        if mapped is not None:
            yield Token(
                mapped,
                str(tok),
                start_pos=tok.start_pos,
                line=tok.line,
                column=tok.column,
                end_line=tok.end_line,
                end_column=tok.end_column,
                end_pos=tok.end_pos,
            )
        else:
            yield tok


class AglLexer(Lexer):
    """Custom Lark lexer for AgL.

    Accepted by the Lark parser via ``lexer=AglLexer``; ``lexer_conf`` is
    received but not used (the grammar's terminal regex patterns are not needed
    because we generate all tokens ourselves).

    ``__future_interface__ = 1`` tells Lark to call ``lex(lexer_state,
    parser_state)`` directly (interface v1), which matches the method
    signature.  Without this, Lark wraps the lexer as interface=0 and calls
    ``lex(text)`` — the wrong arity.
    """

    __future_interface__ = 1

    def __init__(self, lexer_conf: object) -> None:
        # lexer_conf is accepted but unused; the scanner handles all tokenization.
        pass

    def lex(self, lexer_state: LexerState, parser_state: object) -> Iterator[Token]:
        """Tokenize ``lexer_state.text`` and yield Lark tokens.

        Keyword token types are remapped from lowercase (scanner convention)
        to the uppercase names expected by the Lark grammar.
        """
        # lexer_state.text may be a str or a lark.lexer.TextSlice; extract
        # the raw string for the scanner.
        raw = lexer_state.text
        if isinstance(raw, str):
            source = raw
        else:
            # TextSlice: use the underlying .text attribute.
            source = str(raw.text) if hasattr(raw, "text") else str(raw)
        return _remap(layout(scan(source)))
