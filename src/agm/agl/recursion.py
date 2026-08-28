"""The frontend's recursion boundary.

Every static pass walks the syntax tree recursively, so source nested deeply
enough exhausts Python's stack long before the pass finds anything to report.
Without a boundary that failure escapes as an uncaught Python
``RecursionError``, which is a host crash rather than a diagnostic about the
program.

:func:`frontend_recursion_boundary` is the one place that owns that conversion,
mirroring what :meth:`~agm.agl.eval.ir_interpreter.IrInterpreter.run` does for
the evaluator: a ``RecursionError`` raised anywhere inside becomes
:class:`NestingTooDeepError`, an ordinary :class:`~agm.agl.diagnostics.AglError`
that the pipeline and the REPL report through their normal pre-execution
diagnostic channel.  The pipeline wraps each pass it drives — parse, module
load, scope resolution, type check, lowering.  Match compilation needs no
boundary: its recursion follows pattern nesting, which parsing exhausts the
stack on first.

Unlike the evaluator, the frontend does not raise Python's recursion limit.  The
evaluator raises it so that AgL's own ``max_call_depth`` — a language-level
guarantee — is what a recursive program hits first.  The frontend has no such
language-level budget to protect: compilation depth is bounded by the host's
stack either way, and the contract this boundary owns is only that exhausting it
is reported, not crashed on.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from agm.agl.diagnostics import AglError


class NestingTooDeepError(AglError):
    """Source nested too deeply for a static pass to compile it."""


@contextmanager
def frontend_recursion_boundary() -> Iterator[None]:
    """Report a stack-exhausting static pass as an AgL error, not a host crash."""
    try:
        yield
    except RecursionError:
        raise NestingTooDeepError("Source nesting is too deep to compile") from None
