"""The parameter-zone enum shared by ``ir``, ``semantics``, and ``lower``.

``ParamZone`` names which zone (positional-only, standard, named-only) a
parameter belongs to at the IR level. It lives in its own leaf module, apart
from ``ir/program.py``'s data model, so ``semantics`` — which sits below
``ir`` but needs only this enum, not the rest of the execution IR — can
import it without pulling in the IR's function/nominal/program descriptors.

The AST's own zone enum is ``syntax.nodes.ParamKind``; ``typecheck.arguments``
maps one onto the other at the syntax/semantics boundary, keeping this the
single such enum below that boundary.
"""

from __future__ import annotations

import enum

__all__ = ["ParamZone"]


class ParamZone(enum.Enum):
    """The zone a parameter belongs to in its parameter list."""

    POSITIONAL_ONLY = "positional_only"
    STANDARD = "standard"
    NAMED_ONLY = "named_only"
