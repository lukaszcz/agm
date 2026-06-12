"""Shared text helpers for the AgL pipeline.

This is a small, dependency-free leaf module so that *both* the lexer
(``agm.agl.lexer.scanner``) and the evaluator (``agm.agl.eval.interpreter``)
can share the universal-newline normalization without either depending on the
other (the eval pass must never import the lexer — see ``agm/agl/CLAUDE.md``).
"""

from __future__ import annotations


def normalize_newlines(text: str) -> str:
    """Convert CRLF and lone CR line endings to LF (universal-newline style).

    Every ``\\r\\n`` and every lone ``\\r`` becomes a single ``\\n``.  The
    scanner normalizes its source at entry; span offsets index into this
    normalized text, so the evaluator must normalize identically before slicing
    source by offset.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")
