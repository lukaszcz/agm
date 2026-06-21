"""Shared pinned decimal context for all AgL arithmetic.

All AgL evaluation (legacy interpreter and the new IrInterpreter) MUST perform
arithmetic under this context so that results are independent of the host's
ambient ``decimal`` context.

Design (F7): a host that lowers ``getcontext().prec`` would otherwise change
results such as ``1 / 3``.  The context is pinned to 28 significant digits with
ROUND_HALF_EVEN, matching the AgL language specification.
"""

from __future__ import annotations

import decimal

#: Pinned decimal context for all AgL arithmetic.
#:
#: Shared between ``agm.agl.eval.interpreter`` (legacy tree-walker) and
#: ``agm.agl.eval.ir_interpreter`` (new IR evaluator) to guarantee identical
#: numeric semantics without re-deriving the constant.
AGL_DECIMAL_CONTEXT: decimal.Context = decimal.Context(
    prec=28, rounding=decimal.ROUND_HALF_EVEN
)
