"""AgL parser: build a Lark LALR instance and produce ``syntax.Program``.

The module-level ``_PARSER`` is built once at import time from
``grammar/agl.lark`` (loaded via ``importlib.resources`` anchored on
``agm.agl``).

``parse_program(text)`` is the single public entry point.  It feeds the
source string to ``_PARSER``, then passes the resulting Lark tree to
``AstBuilder`` to produce a ``syntax.Program``.  All Lark exceptions and
``LexError``s are wrapped into ``AglSyntaxError``.
"""

from __future__ import annotations

import importlib.resources

from lark import Lark
from lark.exceptions import (
    LarkError,
    UnexpectedCharacters,
    UnexpectedEOF,
    UnexpectedToken,
    VisitError,
)

import agm.agl.syntax as syntax
from agm.agl.lexer.errors import LexError
from agm.agl.lexer.lexer import AglLexer
from agm.agl.parser.errors import AglSyntaxError, syntax_error_from_lark
from agm.agl.parser.transform import AstBuilder


def _load_grammar() -> str:
    """Load the grammar file via importlib.resources (package-anchored)."""
    return (
        importlib.resources.files("agm.agl")
        .joinpath("grammar/agl.lark")
        .read_text(encoding="utf-8")
    )


# Module-level parser instance — built once, reused for every parse call.
_PARSER: Lark = Lark(
    _load_grammar(),
    parser="lalr",
    lexer=AglLexer,
    propagate_positions=True,
    maybe_placeholders=True,
)


def parse_program(text: str, *, filename: str = "<agl>") -> syntax.Program:
    """Parse *text* as an AgL program and return a ``syntax.Program`` AST.

    Parameters
    ----------
    text:
        The source code to parse.
    filename:
        The logical filename for error messages (default ``"<agl>"``).

    Returns
    -------
    syntax.Program
        The root AST node of the parsed program.

    Raises
    ------
    AglSyntaxError
        On any lex or parse error, with a :class:`~agm.agl.syntax.spans.SourceSpan`
        carrying 1-based line/column information.
    """
    try:
        tree = _PARSER.parse(text)
    except LexError as exc:
        raise syntax_error_from_lark(exc, filename=filename) from exc
    except (UnexpectedToken, UnexpectedCharacters, UnexpectedEOF) as exc:
        raise syntax_error_from_lark(exc, filename=filename) from exc
    except LarkError as exc:
        # Any other lark-level error (ParseError, GrammarError, etc.) is a
        # genuine syntax/parse problem.  Narrowing to LarkError lets internal
        # bugs (AssertionError and the like) surface instead of being masked.
        raise syntax_error_from_lark(exc, filename=filename) from exc

    builder = AstBuilder()
    try:
        result = builder.transform(tree)
    except VisitError as exc:
        # Lark wraps transformer exceptions in VisitError.  If the original
        # exception is already an AglSyntaxError, unwrap and re-raise it.
        if isinstance(exc.orig_exc, AglSyntaxError):
            raise exc.orig_exc from exc
        raise syntax_error_from_lark(exc, filename=filename) from exc
    assert isinstance(result, syntax.Program)
    return result
