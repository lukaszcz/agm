"""AgL parser package.

Public API
----------
- :func:`parse_program` — parse and resolve AgL source text into a ``syntax.Program`` AST.
- :func:`parse_program_unresolved` — expose the raw AST before parser-layer infix resolution.
- :func:`wrap_inline_program` — synthesize an inline-source ``program def main`` AST.
- :class:`AglSyntaxError` — span-aware parse error raised on lex/parse failure.

This package is the **only** place in the codebase that imports both ``lark``
and ``agm.agl.syntax``.  Everything downstream of the ``AglSyntaxError`` +
``syntax.Program`` boundary depends only on the AST dataclasses, never on
Lark.  (See ``src/agm/agl/CLAUDE.md`` for the firewall rule.)
"""

from __future__ import annotations

from agm.agl.parser.errors import AglSyntaxError
from agm.agl.parser.parser import (
    has_open_raw_tail_block,
    has_unterminated_triple_quoted_string,
    is_incomplete_source,
    parse_program,
    parse_program_seeded,
    parse_program_unresolved,
    parse_repl_transcript,
    parse_type_expr,
)
from agm.agl.parser.transform import (
    build_infix_operator_table,
    resolve_infix_chains,
    resolve_infix_fixity,
)
from agm.agl.parser.wrap import wrap_inline_program

__all__ = [
    "AglSyntaxError",
    "has_open_raw_tail_block",
    "has_unterminated_triple_quoted_string",
    "is_incomplete_source",
    "parse_program",
    "parse_program_seeded",
    "parse_program_unresolved",
    "parse_repl_transcript",
    "parse_type_expr",
    "build_infix_operator_table",
    "resolve_infix_chains",
    "resolve_infix_fixity",
    "wrap_inline_program",
]
