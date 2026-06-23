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
from dataclasses import replace as dc_replace
from typing import NoReturn

from lark import Lark, Tree
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
from agm.agl.syntax.spans import SourceId


def _reraise_stamped(err: AglSyntaxError, source: SourceId | None) -> NoReturn:
    """Re-raise *err*, stamping its span with *source* when both are present.

    When *source* is not ``None`` and *err* carries a span, a new
    ``AglSyntaxError`` is raised with the span's ``source`` field replaced by
    *source*.  Otherwise *err* is re-raised unchanged.

    This helper consolidates the repeated stamp-then-re-raise pattern that
    appears in every ``except`` arm of :func:`_parse_to_program`.
    """
    if source is not None and err.span is not None:
        raise AglSyntaxError(str(err), span=dc_replace(err.span, source=source)) from err
    raise err


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

# A second Lark instance rooted at the ``type_expr`` grammar rule, built lazily.
# Used only by :func:`parse_type_expr` for the REPL's bare-type-entry fallback;
# keeping a separate start symbol avoids perturbing the program parser's start
# rule.  ``type_expr`` is an existing rule in ``agl.lark`` (every type-expression
# alternative), so this reuses the same grammar/lexer with no grammar changes.
_TYPE_PARSER: Lark | None = None


def _type_parser() -> Lark:
    """Return the lazily-built ``type_expr``-rooted Lark instance."""
    global _TYPE_PARSER
    if _TYPE_PARSER is None:
        _TYPE_PARSER = Lark(
            _load_grammar(),
            parser="lalr",
            lexer=AglLexer,
            propagate_positions=True,
            maybe_placeholders=True,
            start="type_expr",
        )
    return _TYPE_PARSER


def _parse_tree(
    parser: Lark,
    text: str,
    *,
    filename: str,
    source: SourceId | None,
) -> Tree:
    """Parse *text* with *parser*, mapping any lex/parse error to ``AglSyntaxError``.

    Shared by the program parser and the ``type_expr`` parser so the error
    wrapping (and its ``# pragma: no cover`` fallback) exists in one place.
    Returns the raw Lark tree; the caller transforms it.
    """
    try:
        return parser.parse(text)
    except LexError as exc:
        _reraise_stamped(
            syntax_error_from_lark(exc, filename=filename, source_text=text), source
        )
    except (UnexpectedToken, UnexpectedCharacters, UnexpectedEOF) as exc:
        _reraise_stamped(
            syntax_error_from_lark(exc, filename=filename, source_text=text), source
        )
    except LarkError as exc:  # pragma: no cover
        # Any other lark-level error (ParseError, GrammarError, etc.) is a
        # genuine syntax/parse problem.  Narrowing to LarkError lets internal
        # bugs (AssertionError and the like) surface instead of being masked.
        _reraise_stamped(
            syntax_error_from_lark(exc, filename=filename, source_text=text), source
        )


def _transform_tree(
    tree: Tree,
    *,
    start_id: int,
    filename: str,
    source: SourceId | None,
) -> tuple[object, int]:
    """Transform a Lark tree via ``AstBuilder``, unwrapping ``VisitError``.

    Returns ``(result, next_node_id)`` where ``next_node_id`` is the first id NOT
    consumed by the builder's counter (the seed for the next incremental parse).
    """
    builder = AstBuilder(start_id=start_id, source=source)
    try:
        result = builder.transform(tree)
    except VisitError as exc:
        # Lark wraps transformer exceptions in VisitError.  If the original
        # exception is already an AglSyntaxError, unwrap and re-raise it,
        # stamping the source so transformer-raised errors carry the module path.
        if isinstance(exc.orig_exc, AglSyntaxError):
            _reraise_stamped(exc.orig_exc, source)
        raise syntax_error_from_lark(exc, filename=filename) from exc  # pragma: no cover
    return result, builder.next_node_id


def _parse_to_program(
    text: str, *, filename: str, start_id: int, source: SourceId | None = None
) -> tuple[syntax.Program, int]:
    """Parse *text* into a ``Program`` and report the next unused node id.

    Shared body for :func:`parse_program` and :func:`parse_program_seeded`.
    Node ids are assigned starting at *start_id*; the returned ``int`` is the
    first id NOT consumed (the seed for a subsequent incremental parse), read
    from the builder's counter rather than assuming the root holds the maximum.

    When *source* is supplied, every ``SourceSpan`` the builder constructs is
    stamped with that ``SourceId``; the same id is also stamped on any
    ``AglSyntaxError`` raised during parsing.
    """
    tree = _parse_tree(_PARSER, text, filename=filename, source=source)
    result, next_id = _transform_tree(
        tree, start_id=start_id, filename=filename, source=source
    )
    assert isinstance(result, syntax.Program)
    return result, next_id


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
    text: str,
    *,
    filename: str = "<agl>",
    start_id: int = 0,
    source: SourceId | None = None,
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
    source:
        Optional :class:`~agm.agl.syntax.spans.SourceId` to stamp on every
        ``SourceSpan`` in the resulting AST.  When ``None`` (the default),
        spans carry ``UNKNOWN_SOURCE`` (label ``"<agl>"``).  Pass this from
        the module loader so that multi-file diagnostics identify the origin file.

    Returns
    -------
    syntax.Program
        The root AST node of the parsed program.

    Raises
    ------
    AglSyntaxError
        On any lex or parse error, with a :class:`~agm.agl.syntax.spans.SourceSpan`
        carrying 1-based line/column information.  If *source* was supplied,
        the error span is stamped with that ``SourceId``.
    """
    program, _next_id = _parse_to_program(
        text, filename=filename, start_id=start_id, source=source
    )
    return program


def parse_program_seeded(
    text: str,
    *,
    start_id: int,
    filename: str = "<agl>",
    source: SourceId | None = None,
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
    source:
        Optional :class:`~agm.agl.syntax.spans.SourceId` stamped on every span.
        See :func:`parse_program` for details.

    Returns
    -------
    tuple[syntax.Program, int]
        The parsed program and the next unused ``node_id``.

    Raises
    ------
    AglSyntaxError
        On any lex or parse error.
    """
    return _parse_to_program(text, filename=filename, start_id=start_id, source=source)


def parse_type_expr(
    text: str,
    *,
    start_id: int = 0,
    filename: str = "<agl>",
    source: SourceId | None = None,
) -> syntax.TypeExpr:
    """Parse *text* as a single AgL type expression and return a ``TypeExpr``.

    This is a REPL-only convenience: the interactive loop uses it to recognize
    a bare type entry (``int``, ``list[T]``, a declared record/enum name, …) so
    it can echo the resolved type instead of reporting ``'X' is not defined.``.
    It reuses the same grammar/lexer/transformer as :func:`parse_program` but
    roots the parse at the ``type_expr`` rule, so any input that is not a single
    type expression raises ``AglSyntaxError`` and the caller falls back to the
    normal evaluation path.  The language and program parser are unchanged.

    Parameters
    ----------
    text:
        The source code to parse (a single type expression).
    start_id:
        The first ``node_id`` to assign (default ``0``).  Throwaway for the
        REPL fallback, which never promotes; passed for symmetry.
    filename:
        The logical filename for error messages (default ``"<agl>"``).
    source:
        Optional :class:`~agm.agl.syntax.spans.SourceId` stamped on spans.

    Returns
    -------
    syntax.TypeExpr
        The parsed type-expression AST node.

    Raises
    -----
    AglSyntaxError
        On any lex or parse error.
    """
    tree = _parse_tree(_type_parser(), text, filename=filename, source=source)
    result, _next_id = _transform_tree(
        tree, start_id=start_id, filename=filename, source=source
    )
    assert isinstance(result, syntax.TypeExpr)
    return result
