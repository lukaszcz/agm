"""Shared pinned decimal context for all AgL arithmetic.

All AgL evaluation through ``IrInterpreter`` MUST perform
arithmetic under this context so that results are independent of the host's
ambient ``decimal`` context.

Design: a host that lowers ``getcontext().prec`` would otherwise change
results such as ``1 / 3``.  The context is pinned to 28 significant digits with
ROUND_HALF_EVEN, matching the AgL language specification.
"""

from __future__ import annotations

import decimal

#: Pinned decimal context for all AgL arithmetic.
#:
#: Used by ``agm.agl.eval.ir_interpreter`` for stable numeric semantics.
AGL_DECIMAL_CONTEXT: decimal.Context = decimal.Context(prec=28, rounding=decimal.ROUND_HALF_EVEN)
