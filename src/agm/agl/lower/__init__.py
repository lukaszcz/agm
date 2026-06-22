"""AgL lowering phase — compile-time IR emission (M2-A).

Transforms a type-checked ``CheckedProgram`` into an ``ExecutableProgram``
for evaluation.  All implicit coercions are resolved at compile time and
emitted as explicit ``IrCoerce`` nodes; the evaluator switches only on
pre-resolved ``Coercion`` descriptors and never sniffs value types at runtime.

Public API
----------
- :func:`lower_program` — single-module lowering entry point.
- :func:`compile_coercion` — coercion compiler (``Type × Type → Coercion | None``).
"""

from agm.agl.lower.coercions import compile_coercion
from agm.agl.lower.graph import lower_graph
from agm.agl.lower.lowerer import lower_program

__all__ = ["compile_coercion", "lower_graph", "lower_program"]
