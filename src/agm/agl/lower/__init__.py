"""AgL lowering phase — compile-time IR emission.

Transforms a successful ``MatchCompiledProgram`` into an ``ExecutableProgram``
for evaluation. All implicit coercions are resolved at compile time and
emitted as explicit ``IrCoerce`` nodes; the evaluator switches only on
pre-resolved ``Coercion`` descriptors and never sniffs value types at
runtime.

Public API
----------
- :func:`lower_program` — whole-program lowering entry point.
- :func:`lower_repl_program` — incremental whole-program lowering for one
  REPL entry.
- :func:`compile_coercion` — coercion compiler
  (``Type × Type × TypeTable → Coercion | None``).
"""

from agm.agl.lower.coercions import compile_coercion
from agm.agl.lower.program import lower_program
from agm.agl.lower.repl import LinkImage, LoweredReplEntry, ReplPromotionPlan, lower_repl_program

__all__ = [
    "LinkImage",
    "LoweredReplEntry",
    "ReplPromotionPlan",
    "compile_coercion",
    "lower_program",
    "lower_repl_program",
]
