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


def _parse_to_program(
    text: str, *, filename: str, start_id: int
) -> tuple[syntax.Program, int]:
    """Parse *text* into a ``Program`` and report the next unused node id.

    Shared body for :func:`parse_program` and :func:`parse_program_seeded`.
    Node ids are assigned starting at *start_id*; the returned ``int`` is the
    first id NOT consumed (the seed for a subsequent incremental parse), read
    from the builder's counter rather than assuming the root holds the maximum.
    """
    try:
        tree = _PARSER.parse(text)
    except LexError as exc:
        raise syntax_error_from_lark(exc, filename=filename, source_text=text) from exc
    except (UnexpectedToken, UnexpectedCharacters, UnexpectedEOF) as exc:
        raise syntax_error_from_lark(exc, filename=filename, source_text=text) from exc
    except LarkError as exc:  # pragma: no cover
        # Any other lark-level error (ParseError, GrammarError, etc.) is a
        # genuine syntax/parse problem.  Narrowing to LarkError lets internal
        # bugs (AssertionError and the like) surface instead of being masked.
        raise syntax_error_from_lark(exc, filename=filename, source_text=text) from exc

    builder = AstBuilder(start_id=start_id)
    try:
        result = builder.transform(tree)
    except VisitError as exc:
        # Lark wraps transformer exceptions in VisitError.  If the original
        # exception is already an AglSyntaxError, unwrap and re-raise it.
        if isinstance(exc.orig_exc, AglSyntaxError):
            raise exc.orig_exc from exc
        raise syntax_error_from_lark(exc, filename=filename) from exc  # pragma: no cover
    assert isinstance(result, syntax.Program)
    return result, builder.next_node_id


# Single-entry memo for is_incomplete_source: (last_text, last_result).
# The Enter key binding calls is_incomplete_source on each keypress; the
# submit path then parses the identical text a second time.  Caching the most
# recent (text → bool) classification eliminates that redundant parse without
# any risk of stale state — the memo holds at most one entry and is keyed on
# the exact text string.
_incomplete_cache: tuple[str, bool] | None = None


def is_incomplete_source(text: str) -> bool:
    """Return ``True`` when *text* parses as a *prefix* of a valid program.

    This is the structured signal a REPL needs to decide whether pressing Enter
    should submit the entry or insert a continuation newline.  It distinguishes
    "the user has not finished typing" from "the user made a real mistake".

    The classification reads the raw Lark failure rather than the lossy
    ``AglSyntaxError`` message, so it tracks the grammar exactly (this is the one
    place — alongside the rest of the parser package — permitted to import Lark):

    - A clean parse → complete (not incomplete).
    - ``UnexpectedToken`` at the **end of input** (token type ``$END``) → the
      parser ran out of tokens while still expecting more.  This covers every
      unterminated block header (``record R``, ``enum E``, ``case x of``,
      ``try``, ``do agent``, ``if c =>``), a dangling binary operator
      (``1 +``), and an open ``let x =``.  All are treated as "needs more
      input".
    - ``UnexpectedToken`` on a real token where an ``_INDENT`` was expected (the
      user hit Enter right after a block-opening ``=>``/header but has not yet
      indented the suite body) → incomplete.
    - Any other failure (``UnexpectedCharacters``, ``LexError``, a wrong token
      mid-line such as ``let = 5`` or ``x == y``) → complete, so the REPL submits
      and the user sees the genuine error instead of being trapped in a
      continuation prompt.

    Results are memoized for the most recently seen text so that the Enter-key
    check and the immediately following eval-path parse do not trigger two full
    LALR runs for the same source string.
    """
    global _incomplete_cache
    from lark.lexer import Token

    if _incomplete_cache is not None and _incomplete_cache[0] == text:
        return _incomplete_cache[1]

    try:
        _PARSER.parse(text)
        result = False
    except UnexpectedToken as exc:
        # The LALR parser reports a premature end of input as an unexpected
        # ``$END`` token (it never raises ``UnexpectedEOF``), so this single
        # branch classifies every unterminated block / dangling operator.
        token: Token = exc.token
        if token.type == "$END":
            result = True
        elif token.type == "EQ_EQ":
            # ``==`` is always a real error (never a valid token); never a
            # continuation prompt even if ``_INDENT`` appears in expected.
            result = False
        else:
            result = "_INDENT" in exc.expected
    except (LarkError, LexError):
        # Any other parse/lex failure (``UnexpectedCharacters`` and the residual
        # ``LarkError`` family, plus the custom ``LexError`` — which is NOT a
        # ``LarkError`` subclass) is a real error the user should see
        # immediately, not a continuation.
        result = False

    _incomplete_cache = (text, result)
    return result


def parse_program(
    text: str, *, filename: str = "<agl>", start_id: int = 0
) -> syntax.Program:
    """Parse *text* as an AgL program and return a ``syntax.Program`` AST.

    Parameters
    ----------
    text:
        The source code to parse.
    filename:
        The logical filename for error messages (default ``"<agl>"``).
    start_id:
        The first ``node_id`` to assign (default ``0`` → unchanged behaviour).
        Used by incremental sessions to keep node ids globally unique across
        entries; prefer :func:`parse_program_seeded` there to also recover the
        next seed.

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
    program, _next_id = _parse_to_program(text, filename=filename, start_id=start_id)
    return program


def parse_program_seeded(
    text: str, *, start_id: int, filename: str = "<agl>"
) -> tuple[syntax.Program, int]:
    """Parse *text* with node ids starting at *start_id* for incremental use.

    Like :func:`parse_program` but returns ``(program, next_start_id)`` where
    ``next_start_id`` is the first ``node_id`` NOT consumed by this parse — the
    seed to pass as *start_id* for the next entry so that node ids remain
    globally unique across all entries in a REPL session.

    Parameters
    ----------
    text:
        The source code to parse.
    start_id:
        The first ``node_id`` to assign.
    filename:
        The logical filename for error messages (default ``"<agl>"``).

    Returns
    -------
    tuple[syntax.Program, int]
        The parsed program and the next unused ``node_id``.

    Raises
    ------
    AglSyntaxError
        On any lex or parse error.
    """
    return _parse_to_program(text, filename=filename, start_id=start_id)
