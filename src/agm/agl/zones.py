"""The parameter-zone enum, shared across the whole AgL implementation.

``ParamZone`` names which zone (positional-only, standard, named-only) a
parameter belongs to. It is the same fact from the source text down to the
CLI: the transformer assigns it from the ``@arg-*`` attributes an entry or its
declaration carries, shared argument binding enforces it for calls and patterns,
and the host projects a ``program def``'s parameters onto CLI arguments through
it.

It lives in its own dependency-free top-level leaf, alongside
``modules.ids``, so both ends of that span can name it: ``syntax`` (the AST)
and ``ir`` (the execution data model) are deliberately isolated from each
other, and a module either of them may import has to sit below both.
"""

from __future__ import annotations

import enum

__all__ = ["ParamZone"]


class ParamZone(enum.Enum):
    """The zone a parameter belongs to in its parameter list.

    Values are stable strings for debuggability; no code should branch on
    them.
    """

    POSITIONAL_ONLY = "positional_only"
    STANDARD = "standard"
    NAMED_ONLY = "named_only"
