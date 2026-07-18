"""AgL lowering phase — compile-time IR emission.

Transforms a successful ``MatchCompiledModule`` or
``MatchCompiledProgram`` into an ``ExecutableProgram`` for evaluation.
All implicit coercions are resolved at compile time and emitted as explicit
``IrCoerce`` nodes; the evaluator switches only on pre-resolved ``Coercion``
descriptors and never sniffs value types at runtime.

Public API
----------
- :func:`lower_module` — single-module lowering entry point.
- :func:`compile_coercion` — coercion compiler
  (``Type × Type × TypeTable → Coercion | None``).
"""

from agm.agl.lower.coercions import compile_coercion
from agm.agl.lower.lowerer import lower_module
from agm.agl.lower.program import lower_program
from agm.agl.lower.repl import (
    LinkImage,
    LoweredReplEntry,
    lower_repl_entry,
    lower_repl_program,
)

__all__ = [
    "LinkImage",
    "LoweredReplEntry",
    "compile_coercion",
    "lower_program",
    "lower_module",
    "lower_repl_entry",
    "lower_repl_program",
]
