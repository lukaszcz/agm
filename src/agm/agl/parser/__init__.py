"""AgL parser package (Component 2).

Public API
----------
- :func:`parse_program` — parse AgL source text into a ``syntax.Program`` AST.
- :class:`AglSyntaxError` — span-aware parse error raised on lex/parse failure.

This package is the **only** place in the codebase that imports both ``lark``
and ``agm.agl.syntax``.  Everything downstream of the ``AglSyntaxError`` +
``syntax.Program`` boundary depends only on the AST dataclasses, never on
Lark.  (See ``src/agm/agl/CLAUDE.md`` for the firewall rule.)
"""

from __future__ import annotations

from agm.agl.parser.errors import AglSyntaxError
from agm.agl.parser.parser import (
    is_incomplete_source,
    parse_program,
    parse_program_seeded,
)

__all__ = [
    "AglSyntaxError",
    "is_incomplete_source",
    "parse_program",
    "parse_program_seeded",
]
