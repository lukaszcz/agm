"""INDENT / DEDENT filter for the AgL lexer.

Consumes the raw token stream from :mod:`agm.agl.lexer.scanner` and injects
synthetic ``_INDENT`` and ``_DEDENT`` tokens according to the AgL indentation
rules.

Key rules
---------
- The indent stack starts at ``[0]``.
- Inside brackets/parens/braces (``paren_level > 0``) ``_NEWLINE`` tokens are
  suppressed (implicit line continuation).
- On a ``_NEWLINE`` token: compare the carried indentation width against the
  stack top.  Deeper → emit ``_INDENT``.  Same → emit ``_NEWLINE`` as-is.
  Shallower → emit one or more ``_DEDENT`` s and verify alignment.
- Continuation rule: when the first significant token on the next line cannot
  begin a construct — ``|``, ``else``, ``catch``, ``until``, ``done``, ``->``,
  ``=>``, or ``=`` — the ``_NEWLINE`` is suppressed and only the ``_DEDENT`` s
  needed to pop the stack to levels strictly greater than that token's column
  are emitted.  These lines never push an indent.  This is what lets branch
  markers and loop terminators align with the construct that owns them, and
  what lets a signature's return type or a body's ``=`` wrap onto its own
  line.
- Operator continuation: a line break is suppressed outright when it would
  separate an operator from its operand — when the line ends with an operator
  still awaiting one (its indentation is then unmeasured, as inside brackets),
  or when a *more indented* line opens with one.  A trailing ``:=`` is the one
  operator that also opens an indented suite, so a more indented line after it
  opens the assignment's block rather than continuing the line.  The indentation
  requirement on the second form is what keeps a block's own item — ``-x``,
  ``not ready``, ``.map(f)`` — a statement of its own.  Operator position is decided by
  :func:`agm.agl.lexer.operators.stands_as_operator`, the predicate soft-keyword
  promotion later reads, so the two passes cannot disagree.
- At EOF: unwind remaining indent levels with ``_DEDENT`` s.
- Template tokens never produce ``_NEWLINE`` s (the scanner does not emit them
  inside templates).
- Misaligned-dedent diagnostics are positioned at the **first real token** on the
  offending line (i.e. the lookahead ``sig`` token), not at the ``_NEWLINE`` that
  precedes it.  This ensures the reported source line matches the line the user
  actually misindented.
- Synthetic ``done`` injection: when a multi-line loop body is popped without
  an explicit ``until``/``done`` terminator, a synthetic DONE token is emitted
  immediately after the ``_DEDENT``.  This lets the grammar always require a
  terminator while still allowing the omitted-terminator form for multi-line
  loops.  The DONE is *not* injected when the dedent is triggered by an
  ``until``/``done`` token aligned with the enclosing indent level (the
  enclosing level is recorded at the ``_INDENT`` push, not at the ``do`` column,
  so mid-line ``do`` keywords are handled correctly).
"""

from __future__ import annotations

from typing import Iterator

from lark.lexer import Token

from agm.agl.diagnostics import SourceSpan
from agm.agl.keywords import (
    KW_CATCH,
    KW_DO,
    KW_DONE,
    KW_ELSE,
    KW_EXPORT,
    KW_IMPORT,
    KW_UNTIL,
    KW_USE,
)
from agm.agl.lexer.errors import LexError
from agm.agl.lexer.operators import opens_suite, stands_as_operator
from agm.agl.lexer.tokens import (
    ARROW,
    DEDENT,
    EQ,
    INDENT,
    INTERP_END,
    INTERP_START,
    LBRACE,
    LPAR,
    LSQB,
    NEWLINE,
    PIPE,
    RBRACE,
    RPAR,
    RSQB,
    TEMPLATE_END,
    TEMPLATE_START,
    THIN_ARROW,
)

# Tokens that increase paren depth (newlines suppressed inside these)
_OPEN_BRACKETS = {LPAR, LSQB, LBRACE, INTERP_START, TEMPLATE_START}
# Tokens that decrease paren depth
_CLOSE_BRACKETS = {RPAR, RSQB, RBRACE, INTERP_END, TEMPLATE_END}
# Tokens that no construct can begin with: a line opening with one of these
# continues the line before it instead of starting a new item.
_CONTINUATION = {
    PIPE,
    KW_ELSE,
    KW_CATCH,
    KW_UNTIL,
    KW_DONE,
    THIN_ARROW,
    ARROW,
    EQ,
}


# Words that open a module header.  A header spells a path, not an expression,
# so the operator-continuation rules below leave its line alone: the `/*` tail of
# a wildcard header is an operator name that closes an operand, and would
# otherwise read as a dangling operator.
_HEADER_WORDS = frozenset({KW_IMPORT, KW_USE, KW_EXPORT})


def _synthetic(typ: str, value: str, ref: Token) -> Token:
    """Create a synthetic layout token borrowing position from *ref*.

    Position rule (the coherent ruling for layout tokens): synthetic
    ``_NEWLINE``/``_INDENT``/``_DEDENT`` tokens are positioned at the newline
    that introduces the next line — i.e. they borrow the full
    line/column/start_pos/end_pos of the originating ``_NEWLINE`` ``ref`` token
    (which the scanner positions at the ``\\n`` character itself).  At EOF the
    ``ref`` is a synthetic newline anchored at ``len(text)``.
    """
    return Token(
        typ,
        value,
        start_pos=ref.start_pos,
        line=ref.line,
        column=ref.column,
        end_line=ref.end_line,
        end_column=ref.end_column,
        end_pos=ref.end_pos,
    )


def layout(tokens: Iterator[Token]) -> Iterator[Token]:
    """Inject ``_INDENT``/``_DEDENT`` tokens into *tokens*.

    Parameters
    ----------
    tokens:
        Raw token stream from the scanner (contains ``_NEWLINE`` tokens with
        indentation widths as values).

    Yields
    ------
    Token
        The augmented token stream with layout tokens injected.
    """
    indent_stack: list[int] = [0]
    # Parallel stack aligned with indent_stack: each entry is either the
    # enclosing indent level (indent_stack top before the _INDENT push) when a
    # loop body was opened at that indent level, or None if this level is not a
    # loop body.  Seeded with [None] to match indent_stack's initial [0].
    loop_body_enclosing: list[int | None] = [None]
    paren_level = 0

    # One-token lookahead buffer for the branch/compound continuation rule.
    buffered: list[Token] = []
    stream = iter(tokens)

    # Pending loop body flag: set True when a KW_DO token is seen, carried
    # across the bound bracket (if any), cleared on the first non-bound-bracket
    # token on the same line (inline body) or on the _INDENT that opens the body.
    _pending_loop_body: bool = False
    # When we see `do[`, track the paren_level AT which the bound `[` was
    # opened (after incrementing paren_level for that bracket).  Reset to -1
    # when the matching `]` is seen.  Keeps _pending_loop_body alive inside the
    # bound expression so we can still tag the subsequent _INDENT.
    # Note: the `do[` bracket arrives as plain LSQB in the layout pass;
    # DO_LSQB is generated later by _remap_adjacent_brackets, so layout must
    # detect the bound bracket itself using `_pending_loop_body`.
    _do_bound_paren_level: int = -1

    def _next() -> Token | None:
        if buffered:
            return buffered.pop(0)
        try:
            return next(stream)
        except StopIteration:
            return None

    def _peek_pair() -> tuple[Token | None, Token | None]:
        """Peek at the next two tokens without consuming either."""
        first = _next()
        second = _next() if first is not None else None
        for token in (second, first):
            if token is not None:
                buffered.insert(0, token)
        return first, second

    def _pop_level(ref: Token, triggering_tok: Token | None) -> Iterator[Token]:
        """Pop one indent level, emit _DEDENT, and inject synthetic DONE if needed.

        Parameters
        ----------
        ref:
            The _NEWLINE (or synthetic EOF) token used for position borrowing.
        triggering_tok:
            The first significant token that caused the pop (peeked but not consumed),
            or None at EOF.  Used to determine whether this pop is caused by an
            explicit until/done aligned with the do.
        """
        enclosing_col = loop_body_enclosing.pop()
        indent_stack.pop()
        yield _synthetic(DEDENT, "", ref)
        if enclosing_col is not None:
            # This is a loop-body level.  Inject a synthetic DONE unless
            # the triggering token is an explicit until/done whose column
            # equals the enclosing indent level (which means it's the
            # explicit terminator).
            is_explicit_terminator = (
                triggering_tok is not None
                and triggering_tok.type in (KW_UNTIL, KW_DONE)
                and (triggering_tok.column or 1) - 1 == enclosing_col
            )
            if not is_explicit_terminator:
                yield _synthetic(KW_DONE, KW_DONE, ref)

    # Reference token for EOF dedents, and its predecessor: the operator-position
    # test reads the token on either side of a line break.
    last_real: Token | None = None
    prev_real: Token | None = None
    # First significant token of the current logical line, kept across every
    # suppressed newline and cleared when a line break is actually emitted.
    line_start: Token | None = None

    while True:
        tok = _next()
        if tok is None:
            break

        # Track bracket/template depth
        if tok.type in _OPEN_BRACKETS:
            paren_level += 1
            prev_real, last_real = last_real, tok
            line_start = line_start if line_start is not None else tok
            if _pending_loop_body and _do_bound_paren_level == -1:
                # First open bracket on the `do` line (before any body token).
                if tok.type == LSQB:
                    # This [ is the bound bracket of the loop — keep _pending_loop_body
                    # and remember the paren_level at which the bound opened.
                    _do_bound_paren_level = paren_level
                else:
                    # Any other open bracket (LPAR, LBRACE, …) means the body
                    # starts inline.
                    _pending_loop_body = False
            yield tok
            continue

        if tok.type in _CLOSE_BRACKETS:
            if paren_level > 0:
                if _do_bound_paren_level == paren_level and tok.type == RSQB:
                    # Closing the loop-bound bracket.  The bound is done;
                    # the next significant token determines inline vs suite.
                    _do_bound_paren_level = -1
                paren_level -= 1
            prev_real, last_real = last_real, tok
            line_start = line_start if line_start is not None else tok
            yield tok
            continue

        if tok.type != NEWLINE:
            # Track: if we see a non-layout token on the same line as a `do`
            # (after the bound bracket, if any) that is NOT the DO_LSQB itself,
            # the body is inline → clear _pending_loop_body.
            if tok.type == KW_DO:
                # New `do` token: flag that a loop body may follow as a suite.
                _pending_loop_body = True
                _do_bound_paren_level = -1
            elif _pending_loop_body and _do_bound_paren_level == -1:
                # A significant token appeared on the `do` line after the bound
                # (or without a bound) → body is inline, clear pending.
                _pending_loop_body = False
            prev_real, last_real = last_real, tok
            line_start = line_start if line_start is not None else tok
            yield tok
            continue

        # --- _NEWLINE token ---

        # Suppressed inside brackets/templates
        if paren_level > 0:
            continue

        # A newline while _pending_loop_body is set means the loop has a multi-line
        # (suite) body — _do_bound_paren_level is already -1 here because
        # paren_level == 0 (bound is closed) or there was no bound.
        # The _INDENT will push the new level tagged with do_col.
        indent_width = int(str(tok))

        # Peek at the next two tokens: the first for the continuation rules, the
        # second for the operator-position test the leading one applies to it.
        sig, after_sig = _peek_pair()

        in_header = line_start is not None and str(line_start) in _HEADER_WORDS
        current_level = indent_stack[-1]

        # Dangling-operator rule: the line ended with an operator still awaiting
        # its operand, so it is unfinished.  Suppress the _NEWLINE and let the
        # next line continue it at any indentation, exactly as brackets do.  An
        # operator that also opens a suite yields to a more indented line, which
        # opens the block instead.
        if (
            not in_header
            and last_real is not None
            and not (indent_width > current_level and opens_suite(last_real))
            and stands_as_operator(
                last_real,
                prev_real.type if prev_real is not None else None,
                sig.type if sig is not None else None,
            )
        ):
            continue

        if sig is not None and sig.type in _CONTINUATION:
            # Continuation rule: suppress the _NEWLINE and emit only the DEDENTs
            # needed to pop the stack to levels strictly greater than the
            # token's column.  These lines never push an indent.
            kw_col = (sig.column or 1) - 1  # convert 1-based column to 0-based

            while len(indent_stack) > 1 and indent_stack[-1] > kw_col:
                yield from _pop_level(tok, sig)

            # Suppress the _NEWLINE — the token continues the current construct.
            # Clear any _pending_loop_body: the `do` body must have been inline
            # (we already cleared it on the first significant token after `do`).
            _pending_loop_body = False
            continue

        if indent_width > current_level:
            # Leading-operator rule: a more indented line opening with an
            # operator continues the line before it instead of opening a block.
            # Requiring the indentation is what keeps a block's own item — `-x`,
            # `not ready`, `.map(f)` — a statement of its own.
            if (
                not in_header
                and sig is not None
                and stands_as_operator(
                    sig,
                    last_real.type if last_real is not None else None,
                    after_sig.type if after_sig is not None else None,
                )
            ):
                continue
            # Indent: push new level, emit _INDENT.
            # Tag this level with the enclosing indent level if a loop body is being opened.
            indent_stack.append(indent_width)
            tag = current_level if _pending_loop_body else None
            loop_body_enclosing.append(tag)
            _pending_loop_body = False
            line_start = None
            yield _synthetic(INDENT, "", tok)
        elif indent_width == current_level:
            # Same level: emit _NEWLINE as-is
            _pending_loop_body = False
            line_start = None
            yield tok
        else:
            # Dedent: pop levels and emit _DEDENT for each
            while len(indent_stack) > 1 and indent_stack[-1] > indent_width:
                yield from _pop_level(tok, sig)

            if indent_stack[-1] != indent_width:
                # Misaligned dedent — the new indentation is not on the stack.
                # Use the lookahead token (``sig``) to position the diagnostic
                # on the *offending* line rather than the preceding ``_NEWLINE``
                # (which sits on the line before the misaligned content).
                err_ref = sig if sig is not None else tok
                start_off = err_ref.start_pos if err_ref.start_pos is not None else 0
                end_off = err_ref.end_pos if err_ref.end_pos is not None else start_off
                span = SourceSpan(
                    start_line=err_ref.line or 1,
                    start_col=err_ref.column or 1,
                    end_line=err_ref.end_line or err_ref.line or 1,
                    end_col=err_ref.end_column or err_ref.column or 1,
                    start_offset=start_off,
                    end_offset=end_off,
                )
                raise LexError(
                    f"Misaligned dedent: expected indentation {indent_stack[-1]}, "
                    f"got {indent_width}",
                    span=span,
                )
            # Emit _NEWLINE at the restored level
            _pending_loop_body = False
            line_start = None
            yield tok

    # EOF: unwind remaining indent levels with _DEDENT tokens.  ``last_real`` is
    # always set here in practice (reaching this loop body requires a pushed
    # indent, which requires a real token); the fallback only guards the
    # impossible empty-stream case and still carries concrete EOF positions.
    ref = (
        last_real
        if last_real is not None
        else Token(
            NEWLINE,
            "0",
            start_pos=0,
            line=1,
            column=1,
            end_line=1,
            end_column=1,
            end_pos=0,
        )
    )
    while len(indent_stack) > 1:
        yield from _pop_level(ref, None)
